#Importing required libraries.
from p7_dashboard_fonctions_aide import*

#Flask:: App configuration.
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

#Get current directory.
cd = os.getcwd()

#Load Data.
SK_ID_CURR = [1001, 1005, 1006,  1007, 1009]
probabilities = [0.97, 0.96, 0.95, 0.93, 0.58]
dictionary_info = dict(zip(SK_ID_CURR,probabilities))

#Define folder.
FOLDER = os.path.join('static') 
full_path = os.path.join(cd+'/'+FOLDER)

#formulaire d'appel à l'API (facultatif)
class SimpleForm(Form):
    form_id = TextField('id:', validators=[validators.required()])
    @app.route("/", methods=['GET', 'POST'])
    def form():
        form = SimpleForm(request.form)
        print(form.errors)

        if request.method == 'POST':
            form_id=request.form['id']
            print(form_id)
            return(redirect('credit/'+form_id)) 
    
        if form.validate():
            # Save the comment here.
            flash('Vous avez demandé l\'ID : ' + form_id)
            redirect('')
        else:
            flash('Veuillez compléter le champ.')
    
        return render_template('formulaire_id.html', form=form)


#Give name after selection.
@app.route('/credit/<id_client>', methods=['GET'])
def credit(id_client):
    dictionary_data = dictionary_info
    return jsonify(dictionary_data)

#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
    