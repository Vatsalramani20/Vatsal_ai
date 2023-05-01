import cv2


person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

people_count = 0
people_detected = []


min_person_size = 10000
max_person_distance = 100


cap = cv2.VideoCapture(0)

while True:
   
    ret, frame = cap.read()

 
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
    people = person_cascade.detectMultiScale(frame_gray, 1.1, 4)

    
    for (x, y, w, h) in people:
       
        if w * h > min_person_size:
            person_detected = False
      
            for (xp, yp) in people_detected:
                if abs(x - xp) < max_person_distance and abs(y - yp) < max_person_distance:
                    person_detected = True
                    break

            if not person_detected:
                people_count += 1
                people_detected.append((x, y))

   
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  
    cv2.putText(frame, f"Unique people count: {len(people_detected)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


   
    cv2.imshow('People Counting', frame)

 
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
