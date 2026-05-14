import boto3

bucket_name = 'cuschurn-buck' #change to unique
my_region = boto3.Session().region_name
s3 = boto3.resource('s3')
local_file = r"C:\Users\Admin\Documents\customer_churn.csv" # based on file in local directory
s3_file = 'Ingested Data/' + 'customer_churn.csv'
folders = ['Ingested Data/', 'Model 1/']

def launch_bucket():
    if my_region == "us-east-1":
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': my_region}
        )

def data_model_folders(bucket_name_, folders_list):
    for i in folders_list:
        s3.Object(bucket_name_, i).put()

def ingest_data(bucket_name_, local_name, s3_file_name):
    s3.Bucket(bucket_name_).upload_file(local_name, s3_file_name)

def main():
    launch_bucket()
    data_model_folders(bucket_name, folders)
    ingest_data(bucket_name, local_file, s3_file)

if __name__ == '__main__':
    main()
    print(f'AWS S3 Bucket {bucket_name} has been created and '
          'partitioned successfully!')
