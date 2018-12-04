from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from .forms import WebForm
from .generate_report import generate_new
from .modify_database import move_file, read_database
import predict
from predict import _main_
import Classification
from Classification import predict_level
import tensorflow as tf

# Create your views here.


# create graph, session
# with default:
# initialize prediction model
# initialize classification model
graph = tf.Graph()
sess = tf.Session(graph = graph)
with graph.as_default():
	with sess.as_default():
		predict._init_(graph, sess)
		Classification._init_(graph, sess)



def home(request):
	try:
		if request.method == "POST":
			form = WebForm(request.POST, request.FILES or none)
			if form.is_valid():
				print('accepted')
				name = form.cleaned_data['name']
				email = form.cleaned_data['email']
				telno = form.cleaned_data['telno']
				location = form.cleaned_data['location']
				image_pothole = form['image_pothole'].value,
				image_ref1 = form['image_ref1'].value,
				image_ref2 = form['image_ref2'].value,
				form.save()
				# print(image_pothole)
				# generate_new(name, email, telno, location, image_pothole, image_ref1, image_ref2)
				print(name, email, telno, location)
				print('Thank you for your report.')
				mylist = move_file(read_database())
				trigger, crop_image_location = _main_(graph, sess, mylist[5])
				estimated_severity = predict_level(graph, sess, crop_image_location, mylist[5], trigger)
				generate_new(mylist[0], mylist[1], mylist[2], mylist[3], mylist[4], mylist[5], mylist[6], mylist[7], estimated_severity, crop_image_location)

				return render(request, 'C:/Users/User/Desktop/django tutorial/fyp/web/templates/web/thankyou.html')

		print('empty.........................')
		form = WebForm()
		return render(request, 'web/home.html', {'form' : form})
	except:
		print('error')
		form = WebForm()
		return render(request, 'web/thankyou_error.html', {'form' : form})