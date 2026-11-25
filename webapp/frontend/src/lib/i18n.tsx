import { createContext, useContext, useState, type ReactNode } from 'react'

const vi = {
  appName: 'PBL6 · Phân đoạn nhồi máu não',
  loginTitle: 'Đăng nhập',
  loginHint: 'Tài khoản do quản trị viên cấp. Không có đăng ký công khai.',
  username: 'Tên đăng nhập',
  password: 'Mật khẩu',
  signIn: 'Đăng nhập',
  signingIn: 'Đang đăng nhập…',
  signOut: 'Đăng xuất',
  errLogin: 'Sai tên đăng nhập hoặc mật khẩu.',
  errRateLimited: 'Đăng nhập sai quá nhiều lần. Thử lại sau {min} phút.',
  errNetwork: 'Không kết nối được máy chủ.',
  changeTitle: 'Đổi mật khẩu',
  changeForced: 'Đây là lần đăng nhập đầu hoặc mật khẩu vừa được đặt lại. Hãy đặt mật khẩu mới để tiếp tục.',
  currentPassword: 'Mật khẩu hiện tại',
  newPassword: 'Mật khẩu mới',
  confirmPassword: 'Nhập lại mật khẩu mới',
  passwordRule: 'Ít nhất 10 ký tự, khác mật khẩu cũ, không chứa tên đăng nhập.',
  errMismatch: 'Hai mật khẩu mới không khớp.',
  save: 'Lưu',
  saving: 'Đang lưu…',
  roleAdmin: 'Quản trị',
  roleDoctor: 'Bác sĩ',
  rolePatient: 'Bệnh nhân',
  navUsers: 'Tài khoản',
  navAudit: 'Nhật ký',
  navCases: 'Ca bệnh',
  navMyResults: 'Kết quả của tôi',
  usersTitle: 'Quản lý tài khoản',
  createUser: 'Tạo tài khoản',
  displayName: 'Họ tên hiển thị',
  role: 'Vai trò',
  status: 'Trạng thái',
  active: 'Đang hoạt động',
  disabled: 'Đã khóa',
  mustChange: 'Chờ đổi mật khẩu',
  actions: 'Thao tác',
  disable: 'Khóa',
  enable: 'Mở khóa',
  resetPassword: 'Đặt lại mật khẩu',
  tempPasswordFor: 'Mật khẩu tạm cho {user}',
  tempPasswordNote: 'Chỉ hiện một lần. Gửi cho người dùng; họ phải đổi khi đăng nhập lần đầu.',
  copy: 'Sao chép',
  close: 'Đóng',
  confirmReset: 'Đặt lại mật khẩu của {user}? Mọi phiên đăng nhập của họ sẽ bị đăng xuất.',
  auditTitle: 'Nhật ký hệ thống',
  time: 'Thời gian',
  action: 'Hành động',
  outcome: 'Kết quả',
  ip: 'IP',
  you: 'bạn',
  comingSoon: 'Phần này đang được xây dựng (giai đoạn {m}).',
  doctorHome: 'Danh sách ca bệnh',
  patientHome: 'Kết quả của tôi',
  notFound: 'Không tìm thấy trang.',
  forbidden: 'Bạn không có quyền xem trang này.',
  back: 'Về trang chính',
  loading: 'Đang tải…',
}

type Dict = typeof vi

const en: Dict = {
  appName: 'PBL6 · Stroke Lesion Segmentation',
  loginTitle: 'Sign in',
  loginHint: 'Accounts are issued by an administrator. There is no public sign-up.',
  username: 'Username',
  password: 'Password',
  signIn: 'Sign in',
  signingIn: 'Signing in…',
  signOut: 'Sign out',
  errLogin: 'Wrong username or password.',
  errRateLimited: 'Too many failed attempts. Try again in {min} min.',
  errNetwork: 'Cannot reach the server.',
  changeTitle: 'Change password',
  changeForced: 'This is your first sign-in or your password was reset. Set a new password to continue.',
  currentPassword: 'Current password',
  newPassword: 'New password',
  confirmPassword: 'Repeat new password',
  passwordRule: 'At least 10 characters, different from the old one, not containing the username.',
  errMismatch: 'The two new passwords do not match.',
  save: 'Save',
  saving: 'Saving…',
  roleAdmin: 'Admin',
  roleDoctor: 'Doctor',
  rolePatient: 'Patient',
  navUsers: 'Accounts',
  navAudit: 'Audit log',
  navCases: 'Cases',
  navMyResults: 'My results',
  usersTitle: 'Account management',
  createUser: 'Create account',
  displayName: 'Display name',
  role: 'Role',
  status: 'Status',
  active: 'Active',
  disabled: 'Disabled',
  mustChange: 'Must change password',
  actions: 'Actions',
  disable: 'Disable',
  enable: 'Enable',
  resetPassword: 'Reset password',
  tempPasswordFor: 'Temporary password for {user}',
  tempPasswordNote: 'Shown once. Give it to the user; they must change it at first sign-in.',
  copy: 'Copy',
  close: 'Close',
  confirmReset: "Reset {user}'s password? All their sessions will be signed out.",
  auditTitle: 'Audit log',
  time: 'Time',
  action: 'Action',
  outcome: 'Outcome',
  ip: 'IP',
  you: 'you',
  comingSoon: 'This part is being built (milestone {m}).',
  doctorHome: 'Cases',
  patientHome: 'My results',
  notFound: 'Page not found.',
  forbidden: 'You do not have access to this page.',
  back: 'Back to home',
  loading: 'Loading…',
}

export type Lang = 'vi' | 'en'
export type Key = keyof Dict
const DICTS: Record<Lang, Dict> = { vi, en }

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem('pbl6-lang')
    if (saved === 'vi' || saved === 'en') return saved
  } catch { /* storage blocked: use the default */ }
  return 'vi'
}

interface I18n {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: Key, vars?: Record<string, string | number>) => string
}

const Ctx = createContext<I18n | null>(null)

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang)
  const setLang = (l: Lang) => {
    setLangState(l)
    document.documentElement.lang = l
    try { localStorage.setItem('pbl6-lang', l) } catch { /* not persisted */ }
  }
  const t = (key: Key, vars: Record<string, string | number> = {}) =>
    DICTS[lang][key].replace(/\{(\w+)\}/g, (_, k: string) => String(vars[k] ?? ''))
  return <Ctx.Provider value={{ lang, setLang, t }}>{children}</Ctx.Provider>
}

export function useI18n(): I18n {
  const v = useContext(Ctx)
  if (!v) throw new Error('useI18n outside I18nProvider')
  return v
}
