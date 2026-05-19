import yaml
import bcrypt

def get_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

config = {
    'credentials': {
        'usernames': {
            'admin': {
                'email': 'admin@example.com',
                'name': 'Денис (Адмін)',
                'password': get_hash('admin123'),
                'failed_login_attempts': 0,
                'logged_in': False
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'some_signature_key_for_diploma',
        'name': 'bug_classifier_cookie'
    },
    'preauthorized': {
        'emails': []
    }
}

with open('config.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

print("config.yaml successfully generated.")
