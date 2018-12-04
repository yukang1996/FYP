from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, HTML
from django import forms
from django.core.validators import RegexValidator
from .models import Snippet

# class HomeForm(forms.ModelForm):
# 	name = forms.CharField(max_length = 100, validators = [RegexValidator(r'[a-zA-Z]+', 'Please enter a valid name(Only letters)')])
# 	email = forms.EmailField()
# 	telno = forms.CharField(max_length = 12)
# 	location = forms.CharField(max_length = 100)

# 	class Meta:
# 		model = Post
# 		fields = ('name', 'email', 'telno', 'location')

# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, **kwargs)

# 		self.helper = FormHelper
# 		self.helper.form_method = 'post'
# 		self.helper.layout = Layout(
# 			'name',
# 			'email',
# 			'telno',
# 			'location',
# 			Submit('submit', 'Submit', css_class = 'btn-success')
# 		)

# class SnippetForm(forms.ModelForm):

# 	class Meta:
# 		model = Snippet
# 		fields = ('name', 'email', 'telno', 'location', 'document')

class WebForm(forms.ModelForm):
	

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fields['name'].label = "Full Name"
		self.fields['file'].label = "Report File"
		self.fields['admin_name'].label = "Admin Name"
		self.fields['message'].label = "Message"
		self.helper = FormHelper
		self.helper.form_method = 'post'
		self.helper.enctype = 'multipart/form-data'
		self.helper.layout = Layout(
			'name',
			'email',
			'file',
			'message',
			'admin_name',
			Submit('submit', 'Submit', css_class = 'btn-success')

		)
		print('saving...........')

	class Meta:
		model = Snippet
		fields = ('name', 'email', 'file', 'message', 'admin_name')
