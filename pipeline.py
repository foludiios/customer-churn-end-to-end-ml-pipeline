from botocore.exceptions import ClientError
import inception.aws
import src.evaluate
import src.predict
from inception.aws import s3, bucket_name

def bucket_exists():
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError:
        return False

def main():
    if not bucket_exists():
        inception.aws.main()
    src.evaluate.main() # run this, then uvicorn, then ?
    #src.predict.test()

if __name__ == '__main__':
    main()