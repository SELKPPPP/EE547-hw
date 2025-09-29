import argparse, json, os,datetime 
from datetime import datetime, timezone


from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError
import sys
import boto3
from time import sleep

#Required AWS Permissions
"""
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sts:GetCallerIdentity",
                "iam:ListUsers",
                "iam:GetUser",
                "iam:ListAttachedUserPolicies",
                "ec2:DescribeInstances",
                "ec2:DescribeImages",
                "ec2:DescribeSecurityGroups",
                "s3:ListAllMyBuckets",
                "s3:GetBucketLocation",
                "s3:ListBucket"
            ],
            "Resource": "*"
        }
    ]
}
"""

#Network timeouts: Retry once, then skip resource
def retry_once(fn, *args, **kwargs):
    
    try:
        return fn(*args, **kwargs)
    except (EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError):
        sleep(0.8)
        return fn(*args, **kwargs)
    
    
# Part A: identity info

def get_account_info(sess):
    sts = sess.client("sts")
    ident = retry_once(sts.get_caller_identity)
    return {
        "account_id": ident.get("Account"),
        "user_arn": ident.get("Arn"),
        "region": sess.region_name or "N/A",
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
    }

def UTC_time(dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def list_iam_users(user):
    """
    {
    "username": "user-name",
    "user_id": "AIDACKEXAMPLE",
    "arn": "arn:aws:iam::123456789012:user/user-name",
    "create_date": "2025-01-15T10:30:00Z",
    "last_activity": "2025-09-10T14:20:00Z",  # PasswordLastUsed if available
    "attached_policies": [
        {
            "policy_name": "PowerUserAccess",
            "policy_arn": "arn:aws:iam::aws:policy/PowerUserAccess"
        }
    ]
    }
    """
    # Create IAM client
    iam = user.client("iam")

    users_out = []

    # create a paginator to handle large number of users
    try:
        paginator = iam.get_paginator("list_users")
        for page in paginator.paginate():
            for u in page.get("Users", []):
                username = u.get("UserName")

                last_used = None
                try:
                    gu = iam.get_user(UserName=username).get("User", {})
                    last_used = gu.get("PasswordLastUsed")
                except ClientError as e:
                    if e.response["Error"]["Code"] not in ("AccessDenied", "AccessDeniedException"):
                        raise

                # Added policies
                attached_policies = []
                try:
                    ap = retry_once(iam.list_attached_user_policies, UserName=username).get("AttachedPolicies", [])
                    attached_policies = [
                        {"policy_name": p.get("PolicyName"), "policy_arn": p.get("PolicyArn")}
                        for p in ap
                    ]
                except ClientError as e:
                    if e.response["Error"]["Code"] not in ("AccessDenied", "AccessDeniedException"):
                        raise

                users_out.append({
                    "username": username,
                    "user_id": u.get("UserId"),
                    "arn": u.get("Arn"),
                    "create_date": UTC_time(u.get("CreateDate")),
                    "last_activity": UTC_time(last_used),
                    "attached_policies": attached_policies
                })
    except ClientError as e:
        if e.response["Error"]["Code"] in ("AccessDenied", "AccessDeniedException"):
            print("[WARNING] Access denied for IAM operations - skipping user enumeration", file=sys.stderr)
            return []
        raise
    return users_out     



def list_ec2_instances(EC2):
    """
    {
    "instance_id": "i-1234567890abcdef0",
    "instance_type": "t3.micro",
    "state": "running",
    "public_ip": "54.123.45.67",
    "private_ip": "10.0.1.100", 
    "availability_zone": "us-east-1a",
    "launch_time": "2025-09-15T08:00:00Z",
    "ami_id": "ami-0abcdef1234567890",
    "ami_name": "Amazon Linux 2023 AMI",
    "security_groups": ["sg-12345678", "sg-87654321"],
    "tags": {
        "Name": "my-instance",
        "Environment": "development"
    }
    }
    """
    ec2 = EC2.client("ec2")
    instances_out = []

    # create a paginator to handle large number of instances
    try:
        paginator = ec2.get_paginator("describe_instances")
        any_page = False
        for page in paginator.paginate():
            any_page = True
            for res in page.get("Reservations", []):
                for ins in res.get("Instances", []):
                    instances_out.append({
                        "instance_id": ins.get("InstanceId"),
                        "instance_type": ins.get("InstanceType"),
                        "state": ins.get("State", {}).get("Name"),
                        "public_ip": ins.get("PublicIpAddress"),
                        "private_ip": ins.get("PrivateIpAddress"),
                        "availability_zone": ins.get("Placement", {}).get("AvailabilityZone"),
                        "launch_time": UTC_time(ins.get("LaunchTime")),
                        "ami_id": ins.get("ImageId"),
                        "ami_name": ins.get("ImageId"), 
                        "security_groups": [g.get("GroupId") for g in ins.get("SecurityGroups", [])],
                        "tags": {t["Key"]: t["Value"] for t in ins.get("Tags", [])} if ins.get("Tags") else {}
                    })
        if not any_page:
            print(f"[WARNING] No EC2 instances found in {EC2.region_name or 'current region'}", file=sys.stderr)            
    except ClientError as e:                
        if e.response["Error"]["Code"] in ("AccessDenied", "AccessDeniedException"):
            print("[WARNING] Access denied for EC2 operations - skipping instances", file=sys.stderr)
            return []
        raise
    return instances_out


def list_s3_buckets(Buckets):
    """
    {
    "bucket_name": "my-example-bucket",
    "creation_date": "2025-08-20T12:00:00Z",
    "region": "us-east-1",
    "object_count": 47,  # Approximate from ListObjects
    "size_bytes": 1024000  # Approximate total size
    }
    """
    s3 = Buckets.client("s3")
    buckets_out = []

    try:
        resp = retry_once(s3.list_buckets)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("AccessDenied", "AccessDeniedException"):
            print("[WARNING] Access denied for S3 operations - skipping buckets", file=sys.stderr)
            return []
        raise


    for b in resp.get("Buckets", []):
        name = b["Name"]
        creation_date = UTC_time(b.get("CreationDate"))

        #get bucket region
        try:
            loc = s3.get_bucket_location(Bucket=name).get("LocationConstraint")
            region = loc or "us-east-1"   #default us-east-1
        except ClientError as e:
            if e.response["Error"]["Code"] in ("AccessDenied", "AccessDeniedException"):
                print(f"[ERROR] Failed to access S3 bucket '{name}': Access Denied", file=sys.stderr)
                region = "unknown"
            else:
                raise


        #get object count and total size (may be slow for large buckets)
        object_count = 0
        size_bytes = 0
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=name):
                for obj in page.get("Contents", []):
                    object_count += 1
                    size_bytes += obj.get("Size", 0)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("AccessDenied", "AccessDeniedException", "AllAccessDisabled", "NoSuchBucket"):
                print(f"[ERROR] Failed to access S3 bucket '{name}': {e.response['Error']['Message']}", file=sys.stderr)
                object_count, size_bytes = None, None
            else:
                raise

        buckets_out.append({
            "bucket_name": name,
            "creation_date": creation_date,
            "region": region,
            "object_count": object_count,
            "size_bytes": size_bytes
        })

    return buckets_out



#inbound_rules outbound_rules
def format_rule(r, direction="in"):

    proto = r.get("IpProtocol")
    if proto == "-1":
        proto = "all"

    from_port = r.get("FromPort")
    to_port = r.get("ToPort")
    if from_port is None or to_port is None:
        port_range = "all"
    elif from_port == to_port:
        port_range = str(from_port)
    else:
        port_range = f"{from_port}-{to_port}"

    results = []
    key = "source" if direction == "in" else "destination"

    # IPv4
    for ip in r.get("IpRanges", []):
        results.append({
            "protocol": proto,
            "port_range": port_range,
            key: ip.get("CidrIp")
        })
    # IPv6
    for ip in r.get("Ipv6Ranges", []):
        results.append({
            "protocol": proto,
            "port_range": port_range,
            key: ip.get("CidrIpv6")
        })
    # groups
    for g in r.get("UserIdGroupPairs", []):
        results.append({
            "protocol": proto,
            "port_range": port_range,
            key: g.get("GroupId")
        })

    return results


def list_security_groups(security):
    """
   {
    "group_id": "sg-12345678",
    "group_name": "default",
    "description": "Default security group",
    "vpc_id": "vpc-12345678",
    "inbound_rules": [
        {
            "protocol": "tcp",
            "port_range": "22-22",
            "source": "0.0.0.0/0"
        }
    ],
    "outbound_rules": [
        {
            "protocol": "all",
            "port_range": "all",
            "destination": "0.0.0.0/0"
        }
    ]
    }
    """
    ec2 = security.client("ec2")
    sgs_out = []

    try:
        paginator = ec2.get_paginator("describe_security_groups")

        for page in paginator.paginate():
            for g in page.get("SecurityGroups", []):
                inbound = []
                for r in g.get("IpPermissions", []):
                    inbound.extend(format_rule(r, direction="in"))

                outbound = []
                for r in g.get("IpPermissionsEgress", []):
                    outbound.extend(format_rule(r, direction="out"))

                sgs_out.append({
                    "group_id": g.get("GroupId"),
                    "group_name": g.get("GroupName"),
                    "description": g.get("Description"),
                    "vpc_id": g.get("VpcId"),
                    "inbound_rules": inbound,
                    "outbound_rules": outbound
                })

    except ClientError as e:
        if e.response["Error"]["Code"] in ("AccessDenied", "AccessDeniedException"):
            print("[WARNING] Access denied for EC2 security groups - skipping", file=sys.stderr)
            return []
        raise            
    return sgs_out


# Output report

#json report
def build_report(info):
    account = get_account_info(info)
    resources = {
        "iam_users": list_iam_users(info),
        "ec2_instances": list_ec2_instances(info),
        "s3_buckets": list_s3_buckets(info),
        "security_groups": list_security_groups(info),
    }

    running = sum(1 for i in resources["ec2_instances"] if i.get("state") == "running")
    summary = {
        "total_users": len(resources["iam_users"]),
        "running_instances": running,
        "total_buckets": len(resources["s3_buckets"]),
        "security_groups": len(resources["security_groups"]),
    }
    return {"account_info": account, "resources": resources, "summary": summary}


#table report
def print_table(report):
    acc = report["account_info"]
    res = report["resources"]
    out = []
    out.append(f"AWS Account: {acc['account_id']} ({acc['region']})")
    out.append(f"Scan Time: {acc['scan_timestamp'].replace('T',' ')[:-1]} UTC")
    out.append("")
    # IAM
    out.append(f"IAM USERS ({len(res['iam_users'])} total)")
    out.append("Username          Create Date   Last Activity      Policies")
    for u in res["iam_users"]:
        cd = (u["create_date"] or "")[:10]
        la = (u["last_activity"] or "")[:10]
        out.append(f"{u['username']:<16}{cd:<13}{la:<18}{len(u['attached_policies'])}")
    out.append("")


    # EC2
    running = sum(1 for i in res["ec2_instances"] if i.get("state") == "running")
    stopped = sum(1 for i in res["ec2_instances"] if i.get("state") == "stopped")
    out.append(f"EC2 INSTANCES ({running} running, {stopped} stopped)")
    out.append("Instance ID         Type        State     Public IP       Launch Time")
    for i in res["ec2_instances"]:
        lt = (i["launch_time"] or "").replace("T"," ")[:-1][:16]
        out.append(f"{i['instance_id']:<20}{i['instance_type']:<11}{i['state']:<9}{(i['public_ip'] or '-'):<15}{lt}")
    out.append("")


    # S3
    out.append(f"S3 BUCKETS ({len(res['s3_buckets'])} total)")
    out.append("Bucket Name                    Region       Created     Objects   Size (MB)")
    for b in res["s3_buckets"]:
        created = (b["creation_date"] or "")[:10]
        objs = "-" if b["object_count"] is None else str(b["object_count"])
        size_mb = "-" if b["size_bytes"] is None else f"~{round(b['size_bytes']/1024/1024,1)}"
        out.append(f"{b['bucket_name']:<30}{b['region']:<12}{created:<12}{objs:<9}{size_mb}")
    out.append("")


    # SG
    out.append(f"SECURITY GROUPS ({len(res['security_groups'])} total)")
    out.append("Group ID            Name                 VPC ID           Inbound Rules")
    for g in res["security_groups"]:
        out.append(f"{g['group_id']:<20}{g['group_name']:<20}{(g['vpc_id'] or '-'):<16}{len(g['inbound_rules'])}")
    return "\n".join(out)



def session(region_arg=None):
    region = region_arg or os.getenv("AWS_DEFAULT_REGION")
    return boto3.session.Session(region_name=region)


# validate region
def validate_region_or_exit(sess):
    region = sess.region_name
    if region is None:
        return
    valid = set(sess.get_available_regions("ec2"))
    if region not in valid:
        sys.stderr.write(f"[ERROR] Invalid region: {region}\n")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(
        description="AWS Resource Inspector: IAM users, EC2 instances, S3 buckets, and security groups."
    )
    ap.add_argument("--region", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--format", choices=["json", "table"], default="json")
    args = ap.parse_args()

    try:
        sess = session(args.region)
        validate_region_or_exit(sess)
        _ = get_account_info(sess)
    except ClientError as e:
        print(f"[ERROR] Authentication failed: {e.response['Error']['Message']}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}", file=sys.stderr)
        sys.exit(1)

    report = build_report(sess)
    if args.format == "json":
        text = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        text = print_table(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)


if __name__ == "__main__":
    main()