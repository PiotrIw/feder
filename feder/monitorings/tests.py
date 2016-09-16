from django.core.urlresolvers import reverse
from django.test import TestCase

from feder.main.mixins import PermissionStatusMixin
from feder.users.factories import UserFactory
from guardian.shortcuts import assign_perm

from .factories import MonitoringFactory
from .models import Monitoring
from .forms import MonitoringForm

EXAMPLE_DATA = {'name': 'foo-bar-monitoring',
                'description': 'xyz',
                'notify_alert': True,
                'template': 'xyz {{EMAIL}}'}


class MonitoringFormTestCase(TestCase):
    def setUp(self):
        self.user = UserFactory(username="john")

    def test_form_save_user(self):
        form = MonitoringForm(EXAMPLE_DATA.copy(), user=self.user)
        self.assertTrue(form.is_valid(), msg=form.errors)
        obj = form.save()
        self.assertEqual(obj.user, self.user)

    def test_form_template_validator(self):
        data = EXAMPLE_DATA.copy()
        data['template'] = 'xyzyyz'
        form = MonitoringForm(data, user=self.user)
        self.assertFalse(form.is_valid())
        self.assertIn('template', form.errors)


class ObjectMixin(object):
    def setUp(self):
        self.user = UserFactory(username="john")
        self.monitoring = self.permission_object = MonitoringFactory()


class MonitoringCreateViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    url = reverse('monitorings:create')
    permission = ['monitorings.add_monitoring', ]

    def get_permission_object(self):
        return None

    def test_template_used(self):
        assign_perm('monitorings.add_monitoring', self.user)
        self.client.login(username='john', password='pass')
        response = self.client.get(self.get_url())
        self.assertTemplateUsed(response, "monitorings/monitoring_form.html")

    def test_assign_perm_for_creator(self):
        assign_perm('monitorings.add_monitoring', self.user)
        self.client.login(username='john', password='pass')
        data = EXAMPLE_DATA.copy()
        response = self.client.post(self.get_url(), data=data)
        self.assertEqual(response.status_code, 302)
        monitoring = Monitoring.objects.get(name='foo-bar-monitoring')
        self.assertTrue(self.user.has_perm('monitorings.reply', monitoring))


class MonitoringListViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    status_anonymous = 200
    status_no_permission = 200
    permission = []

    def get_url(self):
        return reverse('monitorings:list')

    def test_list_display(self):
        response = self.client.get(self.get_url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.monitoring)


class MonitoringDetailViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    status_anonymous = 200
    status_no_permission = 200
    permission = []

    def get_url(self):
        return self.monitoring.get_absolute_url()

    def test_details_display(self):
        response = self.client.get(self.get_url())
        self.assertContains(response, self.monitoring)


class MonitoringUpdateViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    permission = ['monitorings.change_monitoring', ]

    def get_url(self):
        return reverse('monitorings:update', kwargs={'slug': self.monitoring.slug})

    def test_template_used(self):
        assign_perm('monitorings.change_monitoring', self.user, self.monitoring)
        self.client.login(username='john', password='pass')
        response = self.client.get(self.get_url())
        self.assertTemplateUsed(response, "monitorings/monitoring_form.html")


class MonitoringDeleteViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    permission = ['monitorings.delete_monitoring', ]

    def get_url(self):
        return reverse('monitorings:delete', kwargs={'slug': self.monitoring.slug})

    def test_template_used(self):
        assign_perm('monitorings.delete_monitoring', self.user, self.monitoring)
        self.client.login(username='john', password='pass')
        response = self.client.get(self.get_url())
        self.assertTemplateUsed(response, "monitorings/monitoring_confirm_delete.html")


class PermissionWizardTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    permission = ['monitorings.manage_perm', ]

    def get_url(self):
        return reverse('monitorings:perm-add', kwargs={'slug': self.monitoring.slug})

    def test_template_used(self):
        assign_perm('monitorings.manage_perm', self.user, self.monitoring)
        self.client.login(username='john', password='pass')
        response = self.client.get(self.get_url())
        self.assertTemplateUsed(response, 'monitorings/permission_wizard.html')


class MonitoringPermissionViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    permission = ['monitorings.manage_perm', ]

    def get_url(self):
        return reverse('monitorings:perm', kwargs={'slug': self.monitoring.slug})

    def test_template_used(self):
        assign_perm('monitorings.manage_perm', self.user, self.monitoring)
        self.client.login(username='john', password='pass')
        response = self.client.get(self.get_url())
        self.assertTemplateUsed(response, 'monitorings/monitoring_permissions.html')


class MonitoringUpdatePermissionViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    permission = ['monitorings.manage_perm', ]

    def get_url(self):
        return reverse('monitorings:perm-update', kwargs={'slug': self.monitoring.slug,
                                                          'user_pk': self.user.pk})

    def test_template_used(self):
        assign_perm('monitorings.manage_perm', self.user, self.monitoring)
        self.client.login(username='john', password='pass')
        response = self.client.get(self.get_url())
        self.assertTemplateUsed(response, 'monitorings/monitoring_form.html')
