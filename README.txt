-> Prerequisites

	To run the application, you need to have the following installed:

	Python 3.8 or later
	FastAPI
	Torch
	Torchvision
	Pillow
	Requests

-> You can install the required packages using the following command:

	pip install fastapi torch torchvision pillow requests
	pip install uvicorn

	
-> The repository contains the following files:

	main.py: The main file containing the FastAPI application and the model.
	test_app.py: A script to test the application by sending requests to the API endpoint.
	web_app.py: The web application to interact with the model through a user interface.
	The repository also contains the following directories:

		static: Contains the static files like CSS and JavaScript.
		templates: Contains the HTML templates.
		inputTest : Test Images to see proper working of the application.

-> To run the application :

	Start the main FastAPI application = `uvicorn main:app --host 0.0.0.0 --port 8000`
	
	In another terminal, start the web application: 
	`uvicorn web_app:app --host 0.0.0.0 --port 8001`

-> Go to http://localhost:8001 to access the web application. You can now upload an image 	of a brain MRI scan, and the application will classify it into one of the four 	classes.

	To test the application using the test_app.py script, open another terminal and run 	the following command: python test_app.py

