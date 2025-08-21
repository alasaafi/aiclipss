# create_db.py

from app import app, db, User
from sqlalchemy import inspect, text

def create_db_and_admin():
    with app.app_context():
        inspector = inspect(db.engine)

        # إنشاء الجداول إذا لم تكن موجودة
        if 'user' not in inspector.get_table_names():
            db.create_all()
            print("Database tables created.")
        else:
            print("Database tables already exist.")

        # إضافة عمود 'email' إذا لم يكن موجود
        try:
            columns = [column['name'] for column in inspector.get_columns('user')]
            if 'email' not in columns:
                print("Adding 'email' column to User table...")
                with db.engine.connect() as connection:
                    connection.execute(text('ALTER TABLE user ADD COLUMN email VARCHAR(150) UNIQUE'))
                    connection.commit()
                print("Column 'email' added successfully.")
        except Exception as e:
            print(f"Failed to add email column: {e}")

        # إنشاء المستخدم admin إذا لم يكن موجود
        if not User.query.filter_by(username='admin').first():
            admin_user = User(
                username='admin',
                email='admin@example.com',
                tier='business',
                is_admin=True
            )
            admin_user.set_password('adminpass')
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user 'admin' created with password 'adminpass'")
        else:
            print("Admin user already exists.")

if __name__ == "__main__":
    create_db_and_admin()
