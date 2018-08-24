#!/usr/local/bin/python
import boto3
import botocore
GOES_BUCKET = 'noaa-goes16'
s3=boto3.client('s3')

#response = s3.list_buckets()
#buckets=[bucket['Name'] for bucket in response['Buckets']]
#print("Bucket List: %s" % buckets)	


product='ABI-L2-MCMIPC'
year=2018
day=1
hour=1
prefix='{}/{}/{}/{}/'.format(product,year,'{:03}'.format(day),'{:02}'.format(hour))
response = s3.list_objects_v2(Bucket=GOES_BUCKET,Prefix=prefix)
keys=set([object['Key'] for object in response['Contents']])
print keys
