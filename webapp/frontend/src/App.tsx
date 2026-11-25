import { BrowserRouter, Route, Routes } from 'react-router-dom'

import AppShell from './components/AppShell'
import { AuthProvider } from './lib/auth'
import { I18nProvider } from './lib/i18n'
import AdminAuditPage from './pages/AdminAuditPage'
import AdminUsersPage from './pages/AdminUsersPage'
import ChangePasswordPage from './pages/ChangePasswordPage'
import LoginPage from './pages/LoginPage'
import { Message, Placeholder } from './pages/SimplePages'
import CasePage from './pages/doctor/CasePage'
import PatientPage from './pages/doctor/PatientPage'
import PatientsPage from './pages/doctor/PatientsPage'
import ReviewPage from './pages/doctor/ReviewPage'
import { Guard, HomeRedirect, SignedIn } from './routes'

export default function App() {
  return (
    <I18nProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<HomeRedirect />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/change-password" element={<SignedIn><ChangePasswordPage /></SignedIn>} />
            <Route element={<Guard><AppShell /></Guard>}>
              <Route path="/admin/users" element={<Guard roles={['admin']}><AdminUsersPage /></Guard>} />
              <Route path="/admin/audit" element={<Guard roles={['admin']}><AdminAuditPage /></Guard>} />
              <Route path="/doctor" element={<Guard roles={['doctor']}><PatientsPage /></Guard>} />
              <Route path="/doctor/patients/:id" element={<Guard roles={['doctor']}><PatientPage /></Guard>} />
              <Route path="/doctor/cases/:id" element={<Guard roles={['doctor']}><CasePage /></Guard>} />
              <Route path="/doctor/review" element={<Guard roles={['doctor']}><ReviewPage /></Guard>} />
              <Route path="/patient" element={<Guard roles={['patient']}><Placeholder title="patientHome" milestone="M3" /></Guard>} />
            </Route>
            <Route path="/403" element={<Message text="forbidden" />} />
            <Route path="*" element={<Message text="notFound" />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </I18nProvider>
  )
}
