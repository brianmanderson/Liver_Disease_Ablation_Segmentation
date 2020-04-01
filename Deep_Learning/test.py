from Base_Deeplearning_Code.Send_Email_To_Phone.Send_Email_Module import Send_Email_Class
import os

fid = open(os.path.join('.', 'password.txt'))
line = fid.readline()
fid.close()
data = line.split(',')
email_class_object = Send_Email_Class(email_address=data[0], password=data[1])
email_class_object.set_outbound_email(data[2])
email_class_object.send_email('test')