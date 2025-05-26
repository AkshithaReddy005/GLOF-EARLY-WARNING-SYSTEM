from flask import Flask,render_template,request

from modelpredict import input_model
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        data = request.form
        data_input = data.get('input_string')

        # Debugging: Print the input to the console
        print(f"Received input: {data_input}")

        if not data_input:
            return render_template('index.html', title="GLOF EWS", error="No input provided.")

        # Process the input string using your model function
        results = input_model(data_input)
        
        # Render template with results
        return render_template('index.html', title="GLOF EWS", results=results)
    
    # Render the template for GET requests
    return render_template('index.html', title="GLOF EWS")


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/sign')
def sign():
    return render_template('sign.html')

if __name__ == "__main__":
    app.run(debug=True)
