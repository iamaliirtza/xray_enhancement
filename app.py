from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/enhance', methods=['POST'])
def enhance_image():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Apply enhancements
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img)

    blurred = cv2.GaussianBlur(clahe_image, (5, 5), 1.0)
    unsharp_image = cv2.addWeighted(clahe_image, 1.5, blurred, -0.5, 0)

    f = np.fft.fft2(unsharp_image)
    fshift = np.fft.fftshift(f)
    rows, cols = unsharp_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    img_back = cv2.equalizeHist(img_back)
    
    # Save result
    result_path = 'enhanced_image.jpg'
    cv2.imwrite(result_path, img_back)
    
    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
