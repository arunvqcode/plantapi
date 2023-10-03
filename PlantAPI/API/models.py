from django.db import models

# Create your models here.
class  APICONTENT(models.Model):
    page_no = models.IntegerField(default=0)
    title = models.CharField(max_length=200)
    content = models.TextField()
    url = models.CharField(max_length=200)

    def __str__(self):
        return self.title
    