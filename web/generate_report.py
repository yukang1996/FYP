import os
from fpdf import FPDF
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
from reportlab.lib.pagesizes import landscape, A4
import datetime

def generate_new(id, name, email, telno, location, image_pothole,image_ref1, image_ref2, estimated_severity, crop_image_location):
	image_pothole2 = image_pothole[:-4] + '_detected' + image_pothole[-4:]
	current_time = datetime.datetime.now()
	print(id)
	pdf = canvas.Canvas("./web/Report/%s/Report.pdf"%(id), pagesize = A4)
	pdf.setFont("Helvetica",15)
	pdf.drawCentredString(300, 800,'Pothole Report %a'%(id))
	pdf.drawString(50, 700, 'Pothole Image: ')
	pdf.drawImage(image_pothole2, 200, 550, width = 150, height = 150)
	pdf.drawImage(crop_image_location, 400, 550, width = 150, height = 150)
	pdf.drawString(50,520, 'Estimated Severity: %s'%(estimated_severity))
	pdf.drawString(50,500, 'Reference Image: ')
	pdf.drawImage(image_ref1, 200, 350, 150, 150)
	pdf.drawImage(image_ref2, 400, 350, 150, 150)
	pdf.drawString(50, 320, 'Location: %s'%(location))
	pdf.drawString(50, 200, 'Reported by: ')
	pdf.drawString(50, 150, 'Name: %s'%(name))
	pdf.drawString(50, 130, 'Tel No: %s'%(telno))
	pdf.drawString(50, 110, 'Email: %s'%(email))
	pdf.drawString(50, 90, 'Date Received: %s'%(current_time.strftime("%Y-%m-%d %H:%M")))
	



	pdf.save()
	print('generateddddd')
