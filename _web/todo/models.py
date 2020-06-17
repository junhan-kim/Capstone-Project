from django.db import models

class Todo(models.Model):
    name = models.CharField('Name', max_length=5, blank= True)
    todo = models.CharField('TODO',max_length=50)


    def __str__(self):
        return self.todo

