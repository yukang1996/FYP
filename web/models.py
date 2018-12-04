from django.db import models
from datetime import datetime
import os
# Create your models here.
# class Snippet(models.Model):
# 	name = models.CharField(max_length = 100)
# 	email = models.EmailField()
# 	telno = models.CharField(max_length = 12)
# 	location = models.CharField(max_length = 100)
# 	document = models.ImageField(upload_to = './document/')
# 	uploaded_at = models.DateTimeField(auto_now_add = True)

# 	def __str__(self):
# 		return self.name

def upload_location(file_name):
	location = ''
	filename = os.path.join(location, file_name)
	print(filename)
	# filename = location + datename + '/' + file_name
	return filename



class Snippet(models.Model):
	name = models.CharField(max_length = 100)
	email = models.EmailField()
	telno = models.CharField(max_length = 12)
	location = models.CharField(max_length = 100)
	image_pothole = models.ImageField(upload_to = upload_location('pothole'), blank = True)
	image_ref1 = models.ImageField(upload_to = upload_location('ref'), blank = True)
	image_ref2 = models.ImageField(upload_to = upload_location('ref'), blank = True)


	def __str__(self):
		return self.name
