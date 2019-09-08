
import boto3
# Let's use Amazon S3
#s3 = boto3.resource('s3')
#Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)


import json
#import flask

#from settings import PROJECT_ROOT


if __name__ == "__main__":

    photo='/Users/aarati/Downloads/obama.jpeg'
    #get array of base64, check the first one
    
    client=boto3.client('rekognition')

    with open(photo, 'rb') as image:
        response = client.recognize_celebrities(Image={'Bytes': image.read()})
    print((response))
    print('Detected faces for ' + photo)    
    for celebrity in response['CelebrityFaces']:
        print ('Name: ' + celebrity['Name'])
        '''
        print ('Id: ' + celebrity['Id'])
        print ('Position:')
        print ('   Left: ' + '{:.2f}'.format(celebrity['Face']['BoundingBox']['Height']))
        print ('   Top: ' + '{:.2f}'.format(celebrity['Face']['BoundingBox']['Top']))
        print ('Info')
        
        for url in celebrity['Urls']:
            print ('   ' + url)
        #print
        '''
        if celebrity['Name'] == "Barack Obama":
            print("Yes")
            # get the rest of the frames and send to ML script
'''
    if __name__ == '__main__': #true if you run the script directly
        print("true")
        app.run(debug=True)
'''
