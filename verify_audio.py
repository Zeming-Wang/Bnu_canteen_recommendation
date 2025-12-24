import pyaudio
print("PyAudio imported successfully!")
try:
    p = pyaudio.PyAudio()
    print("PyAudio instance created.")
    info = p.get_host_api_info_by_index(0)
    print("Host API info:", info)
    p.terminate()
except Exception as e:
    print(f"Error initializing PyAudio: {e}")
