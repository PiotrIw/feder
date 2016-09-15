from django.core.urlresolvers import reverse
from django.test import RequestFactory, TestCase

from feder.main.mixins import PermissionStatusMixin
from feder.users.factories import UserFactory

from .factories import CaseFactory
from .views import CaseAutocomplete


class ObjectMixin(object):
    def setUp(self):
        self.user = UserFactory(username="john")
        self.case = CaseFactory()
        self.permission_object = self.case.monitoring


class CaseListViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    status_anonymous = 200
    status_no_permission = 200

    def get_url(self):
        return reverse('cases:list')


class CaseDetailViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    status_anonymous = 200
    status_no_permission = 200

    def get_url(self):
        return reverse('cases:details', kwargs={'slug': self.case.slug})


class CaseCreateViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    permission = ['monitorings.add_case', ]

    def get_url(self):
        return reverse('cases:create', kwargs={'monitoring': str(self.case.monitoring.pk)})


class CaseUpdateViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    permission = ['monitorings.change_case', ]

    def get_url(self):
        return reverse('cases:update', kwargs={'slug': self.case.slug})


class CaseDeleteViewTestCase(ObjectMixin, PermissionStatusMixin, TestCase):
    permission = ['monitorings.delete_case', ]

    def get_url(self):
        return reverse('cases:delete', kwargs={'slug': self.case.slug})


class CaseAutocompleteTestCase(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_filter_by_name(self):
        CaseFactory(name='123')
        CaseFactory(name='456')
        request = self.factory.get('/customer/details', data={'q': '123'})
        response = CaseAutocomplete.as_view()(request)
        self.assertContains(response, '123')
        self.assertNotContains(response, '456')
