Gestures

<table>
<tr>
  <td><b>Gesture</b>b</td><td><b>Action</b></td>
</tr>
<tr>  
  <td>Move index finger</td><td>Move pointer</td>
</tr><tr>    
<td>Quick thumb-index pinch</td><td>Click</td>
  </tr><tr>
<td>Hold pinch for 0.35 seconds</td><td>Start dragging</td>
    </tr><tr>
<td>Release pinch</td><td>Drop/release mouse</td>
      </tr><tr>
<td>Index and middle raised, others folded</td><td>Scroll</td>
        </tr><tr>
<td>Closed fist held for 1 second</td><td>Close active window with Alt+F4</td>
          </tr><tr>
<td>Q</td><td>Stop the program</td>
</tr>
</table>

The Tasks API replaces mp.solutions.hands.Hands() with mp.tasks.vision.HandLandmarker. Unlike the old Solutions API, it requires a .task model file and, in video mode, a monotonically increasing timestamp for every frame.
MODEL:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
