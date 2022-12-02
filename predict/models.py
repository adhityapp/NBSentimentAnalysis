from django.db import models

from operator import mod
from django.db import models


class Post(models.Model):
    judul = models.CharField(max_length=200)
    body = models.TextField()

    def __str__(self):
        return "{}".format(self.judul)
