import os
import sqlite3
from shutil import copy2

def read_database():
	conn = sqlite3.connect(r'C:\Users\User\Desktop\django tutorial\fyp\db.sqlite3')
	c = conn.cursor()
	all_data = list(c.execute('SELECT * FROM web_snippet'))
	print(all_data)
	newest_data = list(c.execute('SELECT * FROM web_snippet ORDER BY id DESC LIMIT 1'))
	print('Newest Data')
	print(newest_data)
	return newest_data

def move_file(newest_data):
	try:
		MEDIA_ROOT = "/Users/User/Desktop/django tutorial/fyp/web/media/"
		file_id = newest_data[0][0]
		name = newest_data[0][1]
		email = newest_data[0][2]
		telno = newest_data[0][3]
		location = newest_data[0][4]
		pothole_image = os.path.join(MEDIA_ROOT, newest_data[0][5])
		image_ref1 = os.path.join(MEDIA_ROOT, newest_data[0][6])
		image_ref2 = os.path.join(MEDIA_ROOT, newest_data[0][7])
		# C:\Users\User\Desktop\django tutorial\fyp\web\Report
		destination_pothole = r"C:\Users\User\Desktop\django tutorial\fyp\web\Report\%s\pothole"%(file_id)
		destination_ref = r"C:\Users\User\Desktop\django tutorial\fyp\web\Report\%s\ref"%(file_id)
		try:
			copy2(pothole_image, destination_pothole)
			copy2(image_ref1, destination_ref)
			copy2(image_ref2, destination_ref)
			print('done copy.')
		except:
			os.makedirs(destination_pothole)
			os.makedirs(destination_ref)
			copy2(pothole_image, destination_pothole)
			copy2(image_ref1, destination_ref)
			copy2(image_ref2, destination_ref)
			print('create file and done copy.')

		
		img_filename = os.listdir(destination_pothole)[0]
		destination_pothole_image = os.path.join(destination_pothole, img_filename)
		print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
		print(destination_pothole_image)

		reference_img1 = os.listdir(destination_ref)[0]
		destination_image_ref1 = os.path.join(destination_ref, reference_img1)
		reference_img2 = os.listdir(destination_ref)[1]
		destination_image_ref2 = os.path.join(destination_ref, reference_img2)

		return file_id, name, email, telno, location, destination_pothole_image, destination_image_ref1, destination_image_ref2
		
	except:
		print("db is empty")

	







newest_data = read_database()
move_file(newest_data)
