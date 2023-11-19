import sqlite3
from flask import Flask, render_template, abort, flash, redirect, request, url_for, render_template
from flask_migrate import Migrate
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
migrate = Migrate()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/user_form')
def new_user():
    return render_template('user.html')

@app.route('/user_info', methods=['POST', 'GET'])
def user_info():
    con = None

    try:
        if request.method == 'POST':
            user_id = request.form['user_id']
            user_password = request.form['user_password']
            user_name = request.form['user_name']

            # Get the file from the request
            user_file = request.files['user_file']

            # Ensure the file is not empty
            if user_file and user_file.filename != '':
                # Securely save the filename
                filename = secure_filename(user_file.filename)

                # Ensure the 'uploads' folder exists
                if not os.path.exists("uploads"):
                    os.makedirs("uploads")

                # Save the file to the uploads folder
                user_file.save(os.path.join("uploads", filename))

                # 데이터베이스 연결
                con = sqlite3.connect("database.db")
                cur = con.cursor()

                # 저장된 파일 경로를 데이터베이스에 저장
                cur.execute("INSERT INTO users (ID, password, file, name) VALUES (?, ?, ?, ?)",
                            (user_id, user_password, os.path.join("uploads", filename), user_name))

                con.commit()  # 변경사항 저장

        msg = "Success"

    except Exception as e:
        print(str(e))
        if con:
            con.rollback()
        msg = "Error: {}".format(str(e))

    finally:
        if con:
            con.close()

    return render_template("result.html", msg=msg)

@app.route('/list')
def list():
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row

    cur = con.cursor()
    cur.execute("select * from users")

    rows = cur.fetchall()
    return render_template("list.html", rows=rows)
    
    
@app.route('/sql_read/<int:myid>')
def sql_read(myid):
    con=sqlite3.connect("database.db")
    con.row_factory=sql.Row
        
    cur=con.cursor()
    cur.excute("select * from users id=?", myid)
        
    rows = cur.fetchall()
    if True:
        return render_template("list.html", rows=rows)
    else:
        return abort(404, "no database")
        
app.run(debug=True)