from django.shortcuts import render
from django.http import HttpResponse
from .forms import WebForm
from django.core.mail import send_mail
# Create your views here.

def webadmin(request):
	form = WebForm(request.POST, request.FILES)
	if form.is_valid():
		print('accepted')
		name = form.cleaned_data['name']
		email = form.cleaned_data['email']
		report = form['file'].value
		message = form.cleaned_data['message']
		admin_name = form.cleaned_data['admin_name']
		form.save()
		print(name, email, report, message, admin_name)
		subject = 'Pothole Report'
		sender = 'jpphb@hotmail.com'
		# send_mail(subject, message, sender, {email}, fail_silently = False)
		return render(request, 'C:/Users/User/Desktop/django tutorial/fyp/webadmin/templates/adminweb/thankyou.html')

	print('no form')
	form = WebForm()
	return render(request, 'adminweb/home.html', {'form' : form})
