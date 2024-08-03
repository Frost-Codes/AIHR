from django.db import models
from django.core.exceptions import ValidationError

# Create your models here.

departments = (('HR', 'Human Resource'), ('IT', 'Information Technology'), ('Marketing', 'Marketing'),
               ('Finance', 'Finance'), ('R&D', 'Research and Development'))


locations = (('Nairobi, Kenya', 'Nairobi, Kenya'), ('California, US', 'California, US'),
             ('London, UK', 'London, UK'), ('San fransisco, US', 'San fransisco, US'),
             ('Cape Town, SA', 'Cape Town, SA'))

genders = (('Male', 'Male'), ('Female', 'Female'), ('Prefer not to say', 'Prefer not to say'))


def validate_pdf(value):
    if not value.name.endswith('.pdf'):
        raise ValidationError('Only PDF files are allowed as CV')


class Job(models.Model):
    job_title = models.CharField(max_length=100, null=False)
    department = models.CharField(max_length=50, choices=departments)
    salary = models.IntegerField()
    location = models.CharField(max_length=100, choices=locations)
    introduction = models.CharField(max_length=220)
    brief_posting_description = models.CharField(max_length=400)
    responsibilities = models.CharField(max_length=3000)
    qualifications = models.CharField(max_length=1500)

    is_active = models.BooleanField(default=True)
    date_posted = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.job_title

# job = Job.object.get(id=1)
# job.applicants


class Applicant(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    gender = models.CharField(max_length=20, choices=genders)
    email = models.EmailField()
    location = models.CharField(max_length=100, choices=locations)
    cv = models.FileField(upload_to='cvs/', validators=[validate_pdf])
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='applicants')

    def __str__(self):
        return self.first_name


class Shortlist(models.Model):
    applicant = models.ForeignKey(Applicant, on_delete=models.CASCADE, related_name='shortlisted_jobs')
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='shortlist')
    score = models.IntegerField()
    summary = models.CharField(max_length=1000)

    def __str__(self):
        return self.applicant.first_name
