import string
import bcrypt
import base64  # 新增：用于图片转Base64
import cv2  # 新增：用于图像标注
from flask import Flask, redirect, render_template, url_for, request
from markupsafe import Markup
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------
# （模型加载代码不变，此处省略，保持原有逻辑）
# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Loading plant disease classification model
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

# disease prediction
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# -------------------------新增：图像病害区域标注函数 -----------------------------------------------
def annotate_disease_region(img_bytes, disease_label):
    """
    针对指定病害列表优化的标注函数（颜色阈值+边缘检测，演示级精度）
    :param img_bytes: 图片字节数据
    :param disease_label: 预测的病害标签（如Apple___Apple_scab）
    :return: 标注后的Base64图片字符串
    """
    # 1. 转换为OpenCV格式
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # 2. 仅对非健康样本标注病害区域
    if 'healthy' not in disease_label:
        # ---------------------- 按病害类型定制参数 ----------------------
        disease_config = {
            'Apple___': {
                'hsv_lower': np.array([0, 40, 30]),
                'hsv_upper': np.array([30, 255, 200]),
                'canny_low': 30,
                'canny_high': 120,
                'min_area': 30
            },
            'Cherry___Powdery_mildew': {
                'hsv_lower': np.array([0, 0, 180]),
                'hsv_upper': np.array([180, 50, 255]),
                'canny_low': 20,
                'canny_high': 100,
                'min_area': 20
            },
            'Corn___': {
                'hsv_lower': np.array([0, 60, 50]),
                'hsv_upper': np.array([20, 255, 200]),
                'canny_low': 40,
                'canny_high': 140,
                'min_area': 40
            },
            'Grape___': {
                'hsv_lower': np.array([0, 80, 20]),
                'hsv_upper': np.array([20, 255, 150]),
                'canny_low': 35,
                'canny_high': 130,
                'min_area': 35
            },
            'Orange___': {
                'hsv_lower': np.array([20, 50, 100]),
                'hsv_upper': np.array([40, 255, 200]),
                'canny_low': 25,
                'canny_high': 110,
                'min_area': 50
            },
            'Peach___Bacterial_spot': {
                'hsv_lower': np.array([0, 50, 40]),
                'hsv_upper': np.array([15, 255, 180]),
                'canny_low': 30,
                'canny_high': 120,
                'min_area': 15
            },
            'Pepper,_bell___Bacterial_spot': {
                'hsv_lower': np.array([0, 60, 30]),
                'hsv_upper': np.array([20, 255, 180]),
                'canny_low': 35,
                'canny_high': 130,
                'min_area': 20
            },
            'Potato___': {
                'hsv_lower': np.array([0, 70, 20]),
                'hsv_upper': np.array([25, 255, 180]),
                'canny_low': 45,
                'canny_high': 150,
                'min_area': 45
            },
            'Squash___Powdery_mildew': {
                'hsv_lower': np.array([0, 0, 180]),
                'hsv_upper': np.array([180, 50, 255]),
                'canny_low': 20,
                'canny_high': 100,
                'min_area': 20
            },
            'Strawberry___Leaf_scorch': {
                'hsv_lower': np.array([0, 50, 30]),
                'hsv_upper': np.array([20, 255, 150]),
                'canny_low': 30,
                'canny_high': 120,
                'min_area': 30
            },
            'Tomato___': {
                'hsv_lower': np.array([0, 60, 30]),
                'hsv_upper': np.array([25, 255, 180]),
                'canny_low': 40,
                'canny_high': 140,
                'min_area': 25
            }
        }

        # 匹配当前病害的配置参数
        config = None
        for key in disease_config.keys():
            if key in disease_label:
                config = disease_config[key]
                break
        # 默认配置（匹配不到时使用）
        if config is None:
            config = {
                'hsv_lower': np.array([0, 40, 30]),
                'hsv_upper': np.array([30, 255, 200]),
                'canny_low': 35,
                'canny_high': 130,
                'min_area': 30
            }

        # ---------------------- 核心标注逻辑 ----------------------
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        disease_mask = cv2.inRange(hsv, config['hsv_lower'], config['hsv_upper'])
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        non_green_mask = cv2.bitwise_not(green_mask)
        final_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=non_green_mask)

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        # 灰度化 + 边缘检测
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=final_mask)
        edges = cv2.Canny(gray_masked, config['canny_low'], config['canny_high'])

        # 查找轮廓并绘制红色框
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > config['min_area'] and area < (h * w) / 5:
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                cv2.rectangle(cv_img, (x, y), (x + w_cnt, y + h_cnt), (0, 0, 255), 2)
                cv2.putText(cv_img, "Disease", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 3. 转换回PIL格式并编码为Base64
    annotated_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    annotated_img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/jpeg;base64,{img_base64}"


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    # Get predictions from model
    yb = model(img_u)
    # Normalize probabilities using softmax
    probabilities = torch.nn.functional.softmax(yb, dim=1)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    confidence = probabilities[0, preds[0]].item()
    # Return prediction and confidence
    return prediction, confidence


# -------------------------核心修复：调整Flask应用和数据库初始化顺序 -------------------------
# 1. 先创建Flask应用实例
app = Flask(__name__)

# 2. 再配置数据库和密钥（关键！必须在初始化db之前配置）
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SECRET_KEY"] = 'thisissecretkey'
# 可选：关闭SQLAlchemy的修改跟踪，提升性能
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 3. 初始化数据库、Bcrypt、LoginManager（现在配置已生效）
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# -------------------------数据库模型定义（必须在db初始化后）-------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class UserAdmin(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class ContactUs(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(500), nullable=False)
    text = db.Column(db.String(900), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.title}"


# -------------------------表单定义-------------------------
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "username"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "password"})
    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError("That username already exist. please choose different one.")


class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "username"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "password"})
    submit = SubmitField("Login")


# -------------------------登录管理器回调-------------------------
@login_manager.user_loader
def load_user(user_id):
    # 修复：同时支持普通用户和管理员登录（根据ID查询两个表）
    user = User.query.get(int(user_id))
    if user:
        return user
    return UserAdmin.query.get(int(user_id))


# -------------------------路由定义-------------------------
@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        text = request.form['text']
        contacts = ContactUs(name=name, email=email, text=text)
        db.session.add(contacts)
        db.session.commit()

    return render_template("contact.html")


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    elif form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))

    return render_template("login.html", form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    title = 'dashboard'
    return render_template('dashboard.html', title=title)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('hello_world'))


@app.route("/signup", methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template("signup.html", form=form)


@app.route('/crop-recommend')
@login_required
def crop_recommend():
    title = 'crop-recommend - Crop Recommendation'
    return render_template('crop.html', title=title)


@app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = '- Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)


# 在 disease_prediction 路由中修复变量传递
@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required
def disease_prediction():
    title = '- Disease Detection'
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            # 1.读取图片并预测病害
            img = file.read()
            prediction, confidence = predict_image(img)

            # 2.解析病害信息 - 使用新的数据结构
            disease_info = disease_dic.get(prediction, {
                'crop_name': '未知作物',
                'disease_name': '未知病害',
                'reasons': ['无法获取病害原因信息'],
                'treatments': ['无法获取防治建议']
            })

            crop_name = disease_info['crop_name']
            disease_name = disease_info['disease_name']
            reasons = disease_info['reasons']
            treatments = disease_info['treatments']

            # 3.标注病害区域并转为Base64
            annotated_img_base64 = annotate_disease_region(img, prediction)

            # 4.传递所有必要的变量到结果页
            return render_template('disease-result.html',
                                   crop_name=crop_name,
                                   disease_name=disease_name,
                                   reasons=reasons,
                                   treatments=treatments,
                                   confidence=confidence,
                                   annotated_img=annotated_img_base64,
                                   title=title)
        except Exception as e:
            print(f"Error: {e}")
            return render_template('disease.html', title=title, error="图片处理失败，请重试！")
    return render_template('disease.html', title=title)

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = '- Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = '- Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        key = 'NHigh' if n < 0 else "Nlow"
    elif max_value == "P":
        key = 'PHigh' if p < 0 else "Plow"
    else:
        key = 'KHigh' if k < 0 else "Klow"

    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)


@app.route("/display")
def querydisplay():
    alltodo = ContactUs.query.all()
    return render_template("display.html", alltodo=alltodo)


@app.route("/AdminLogin", methods=['GET', 'POST'])
def AdminLogin():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for('admindashboard'))

    elif form.validate_on_submit():
        user = UserAdmin.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('admindashboard'))

    return render_template("adminlogin.html", form=form)


@app.route("/admindashboard")
@login_required
def admindashboard():
    alltodo = ContactUs.query.all()
    alluser = User.query.all()
    return render_template("admindashboard.html", alltodo=alltodo, alluser=alluser)


@app.route("/reg", methods=['GET', 'POST'])
def reg():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = UserAdmin(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('AdminLogin'))

    return render_template("reg.html", form=form)


# -------------------------创建数据库表（关键！）-------------------------
with app.app_context():
    db.create_all()  # 在应用上下文中创建所有表

if __name__ == "__main__":
    app.run(debug=True, port=8000)