import cv2
import numpy as np
import math

#*********************DOUGLAS PEUCKER******************************************

def douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points

    start = points[0]
    end = points[-1]

    max_dist = 0
    index = 0
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i

    if max_dist > epsilon:
        left = douglas_peucker(points[:index+1], epsilon)
        right = douglas_peucker(points[index:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])




def perpendicular_distance(point, line_start, line_end):
    if np.all(line_start == line_end):
        return np.linalg.norm(point - line_start)
    return abs(np.cross(line_end - line_start, line_start - point) / np.linalg.norm(line_end - line_start))

# =============================================================================
# ###DEBUG ICIN
# def detect_shape(cnt, debug_img):
#     area = cv2.contourArea(cnt)
#     perimeter = cv2.arcLength(cnt, True)
#     if area == 0 or perimeter == 0:
#         return "Bilinmeyen"
# 
#     circularity = (4 * math.pi * area) / (perimeter ** 2)
#     if circularity > 0.85:
#         return "Daire"
# 
#     # Debug: Farklı epsilon değerlerini dene
#     for eps in range(1 , 21):  
#         simplified = douglas_peucker(cnt.squeeze(), eps)
#         
#         print(f"Epsilon: {eps}, Köşe Sayısı: {len(simplified)}")
#         
#         for point in simplified:
#             cv2.circle(debug_img, tuple(point.astype(int)), 5, (0,0,255), -1)
#         cv2.putText(debug_img, f"eps:{eps}", 
#                     tuple(simplified[0].astype(int)), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
# 
#     return "Kontrol Ediliyor"
# 
# 
# =============================================================================

#**************************DETECT SHAPE*************************************

def detect_shape(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if area == 0 or perimeter == 0:
        return "Bilinmeyen Sekil"

    
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    if circularity > 0.80:
        return "Daire"

    print(circularity)


    
    epsilon = 0.02 * perimeter #DOUGLAS PEUCKER EPSILON
    simplified = douglas_peucker(cnt.squeeze(), epsilon)    
    simplified = remove_duplicate_points(simplified)        ###BİRLEŞİM NOKTALARINI AYNI SAYIYOR FAZLA KÖŞE ÇIKIYOR
    simplified = remove_flat_angles(simplified, angle_threshold=160)        ###160 DERECEDEN BÜYÜK AÇILARI SİLİYOR
    print(simplified)
    corners = len(simplified)
    
    
    ratio = (perimeter ** 2) / area
    
### KÖŞE SAYISINA GÖRE ŞEKİL TAHMİNİ
    if corners == 3:
        return "Ucgen (Kose Sayisi)"
    elif corners == 4:
        return "Dortgen (Kose Sayisi)"
    elif corners == 5:
        return "Besgen (Kose Sayisi)"
    elif corners == 6:
        return "Altigen (Kose Sayisi)"
    else:
        print('corner fazla!!! ratio bakilacak : ', corners)
        
        if ratio < 16:
            print('ratio : ', ratio)
            return "Ucgen (Ratio)"
        elif 16 <= ratio < 18:
            print('ratio : ', ratio)
            return "Dortgen (Ratio)"
        elif 18 <= ratio < 20:
            print('ratio : ', ratio)
            return "Besgen (Ratio)"
        elif 20 <= ratio < 23:
            print('ratio : ', ratio)
            return "Altigen (Ratio)"
        else:
            print('ratio : ', ratio)
            return "Bilinmeyen"
        
        
        
        
#****************DUPLICATE POINTS***********************    
   #BİRLEŞİM NOKTALARINI 2 NOKTA ŞEKLİNDE SAYDIĞI İÇİN NOKTA TEMZİZLİĞİ     
def remove_duplicate_points(points, tol=50):
    cleaned = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - cleaned[-1]) > tol:
            cleaned.append(p)
            
            
    if np.linalg.norm(cleaned[0] - cleaned[-1]) < tol:
        cleaned.pop()
    return np.array(cleaned)
        
        
        

#**************FLAT ANGLES**************************************
### GEREKSİZ NOKTALARI AÇI TESPİTİ İLE TEMİZLEME


def remove_flat_angles(points, angle_threshold):
    
    def calc_angle(a, b, c):
        ba = a - b
        bc = c - b
        
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))   ###cos açıcısını bulduk
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))     ### -1 +1 sınırlandı
        
        return np.degrees(angle_rad)
    
    filtered = []
    
    
    print("\n Açı Kontrolü:")
    for i in range(len(points)):
        prev = points[i -1]
        curr = points[i]
        next = points[(i + 1) % len(points)]
        
        
        angle = calc_angle(prev, curr, next)
        print(f"  Köşe {i}: Açı = {angle:.2f}°", end='')
        
        if angle < angle_threshold:
            filtered.append(curr)
            print(" (kabul edildi)")

        else:
            print(" (elendi - düz köşe)")
            
            
    print(f"---- Toplam kabul edilen köşe: {len(filtered)}\n")
    return np.array(filtered)
        
        
    
    
        


##################################################################
#*******************PREPROCESS*************************************
image = cv2.imread('test2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gauss = cv2.GaussianBlur(gray, (3, 3), 0)
median = cv2.medianBlur(gauss, 5)
cv2.imwrite('1median.png', median)

# =============================================================================
# thresh = cv2.adaptiveThreshold(median, 255,
#                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY_INV, 11, 3)
# =============================================================================


_, thresh = cv2.threshold(median, 110, 255, cv2.THRESH_BINARY)  #retval_
cv2.imwrite('2thresh.png', thresh)


thresh = cv2.bitwise_not(thresh)
cv2.imwrite('3bitwise.png', thresh)


# =============================================================================
# kernel = np.ones((5, 5), np.uint8)
# clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# kernel = np.ones((15, 15), np.uint8)
# clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
# =============================================================================
# =============================================================================
# cv2.imwrite('4kernel.png', clean)
# =============================================================================


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))


MIN_AREA = 1000
debug_image = image.copy()

for cnt in contours:
    if cv2.contourArea(cnt) < MIN_AREA:
        continue

    shape_name = detect_shape(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(image, shape_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


#########################################################################################################

#**************************IMSHOW*************************************************

max_width = 1000
max_height = 800
height, width = image.shape[:2]
scale = min(max_width / width, max_height / height)
resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))

    


while True:
    cv2.imshow("Sekiller", resized_image)

    key = cv2.waitKey(1) & 0xFF  

    if key == ord('q'):
        break

cv2.destroyAllWindows()