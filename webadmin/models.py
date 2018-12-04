from django.db import models
import os
# Create your models here.

def upload_location(file_name):
	location = ''
	filename = os.path.join(location, file_name)
	print(filename)
	# filename = location + datename + '/' + file_name
	return filename

class Snippet(models.Model):
	name = models.CharField(max_length = 100)
	email = models.EmailField()
	file = models.FileField(blank = True)
	message = models.TextField(max_length = 300)
	admin_name = models.CharField(max_length = 100)


	def __str__(self):
		return self.name
