#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response, url_for
from flask_httpauth import HTTPBasicAuth
from flask_cors import CORS
import run_with_mask

from flask_socketio import send, emit, SocketIO

import base64
import io
import cv2
from imageio import imread
import json
import matplotlib.pyplot as plt
import sys
import algoritmi as alg

# sys.setrecursionlimit(3000)

app = Flask(__name__, static_url_path="")
auth = HTTPBasicAuth()

socketio = SocketIO(app, cors_allowed_origins='*', async_handlers=True)

CORS(app)


@socketio.on('inpaint')
def handle_my_custom_event(json1):
    json_object = json.loads(json1)
    img = imread(io.BytesIO(base64.b64decode(json_object['title'])))
    mask = imread(io.BytesIO(base64.b64decode(json_object['description'])))
    metoda = json_object['done']

    for arrays in mask:
        for pixel in arrays:
            if pixel[3] != 0:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255
            pixel[3] = 255

    from PIL import Image
    im = Image.fromarray(img)
    im.save("saved_image.png")

    im2 = Image.fromarray(mask)
    im2.save("saved_mask.png")

    img = cv2.imread('saved_image.png')
    mask = cv2.imread('saved_mask.png', 0)

    if metoda == "1":
        dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    else:
        dst = alg.final_algorithm(img, mask, 15)

    cv2.imwrite('result.png', dst)

    socketio.sleep(0)
    with open("result.png", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
    socketio.emit('inpainted_image', b64_string.decode("utf-8"))
    # sio.emit('my response', {'my response': 'my response'})
    print("emitted")


@auth.get_password
def get_password(username):
    if username == 'miguel':
        return 'python'
    return None


@auth.error_handler
def unauthorized():
    # return 403 instead of 401 to prevent browsers from displaying the default
    # auth dialog
    return make_response(jsonify({'error': 'Unauthorized access'}), 403)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]


def make_public_task(task):
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id=task['id'],
                                      _external=True)
        else:
            new_task[field] = task[field]
    return new_task


@app.route('/todo/api/v1.0/tasks', methods=['GET'])
# @auth.login_required
def get_tasks():
    return jsonify({'tasks': [make_public_task(task) for task in tasks]})


@app.route('/todo/api/v1.0/inpainted_image', methods=['GET'])
# @auth.login_required
def get_inpainted_image():
    with open("result.png", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
    output = {
        "image": b64_string
    }
    # return json.dumps(output)
    return jsonify({'image': b64_string.decode("utf-8")})


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
@auth.login_required
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': make_public_task(task[0])})


@app.route('/todo/api/v1.0/masks', methods=['GET'])
# @auth.login_required
def get_masks():
    print(2002)
    return run_with_mask.get_masks('elephant.jpg')


@app.route('/todo/api/v1.0/put_masks', methods=['POST'])
def create_masks():
    img = imread(io.BytesIO(base64.b64decode(request.json['title'])))
    from PIL import Image
    im = Image.fromarray(img)
    im.save("saved_image_for_ai.png")

    json = run_with_mask.get_masks("saved_image_for_ai.png")
    socketio.emit('ai_masks', json)

    return json


@app.route('/todo/api/v1.0/tasks', methods=['POST'])
# @auth.login_required
def create_task():
    print(1)
    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(request.json['title'])))
    mask = imread(io.BytesIO(base64.b64decode(request.json['description'])))
    metoda = request.json['done']

    for arrays in mask:
        for pixel in arrays:
            if pixel[3] != 0:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255
            pixel[3] = 255

    from PIL import Image
    im = Image.fromarray(img)
    im.save("saved_image.png")

    im2 = Image.fromarray(mask)
    im2.save("saved_mask.png")

    img = cv2.imread('saved_image.png')
    mask = cv2.imread('saved_mask.png', 0)

    if metoda == "1":
        dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    else:
        height = img.shape(0)
        width = img.shape(1)
        dim = height + width // 200
        print(dim)
        dst = alg.final_algorithm(img, mask, dim)

    cv2.imwrite('result.png', dst)

    socketio.sleep(0)
    with open("result.png", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
    socketio.emit('inpainted_image', b64_string.decode("utf-8"))
    # sio.emit('my response', {'my response': 'my response'})
    print("emitted")

    # show image
    # plt.figure()
    # plt.imshow(img, cmap="gray")
    if not request.json or 'title' not in request.json:
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1 if len(tasks) > 0 else 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': make_public_task(task)}), 201


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['PUT'])
@auth.login_required
def update_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and \
            not isinstance(request.json['title'], str):
        abort(400)
    if 'description' in request.json and \
            not isinstance(request.json['description'], str):
        abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description',
                                              task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify({'task': make_public_task(task[0])})


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['DELETE'])
@auth.login_required
def delete_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    tasks.remove(task[0])
    return jsonify({'result': True})


if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app)

# img = imread(io.BytesIO(base64.b64decode(b64_string)))
