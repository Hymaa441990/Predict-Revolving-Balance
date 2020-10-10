import pandas as pd
from flask import Flask, request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('C:/Hymaa/Data Science/Project 1/Deployment/model.pkl','rb'))

cols_when_model_builds = model.get_booster().feature_names

@app.route('/')
def index():
    return render_template('RevolvingBalance.html')

@app.route('/',methods=['POST'])
def Get_All_Details():    
    if request.method == 'POST':
          print('post')
          loan_amnt = request.form.getlist("loan_amnt")
          annual_inc = request.form.getlist("annual_inc")
          debt_income_ratio = request.form.getlist("debt_income_ratio")
          numb_credit = request.form.getlist("numb_credit")
          Rate_of_intrst = request.form.getlist("Rate_of_intrst")
          tot_curr_bal = request.form.getlist("tot_curr_bal")
          total_credits = request.form.getlist("total_credits")
          total_rec_int = request.form.getlist("total_rec_int")
          home_ownership = request.form.getlist("home_ownership")
          verification_status = request.form.getlist("verification_status")
          initial_list_status = request.form.getlist("initial_list_status")
          Experience = request.form.getlist("Experience")
          purpose = request.form.getlist("purpose")  
          
          print(tot_curr_bal[0])
          print(Rate_of_intrst[0])
          print(annual_inc[0])
          print(debt_income_ratio[0])
          ANY=0
          MORTGAGE=0
          NONE=0
          OTHER=0
          OWN=0
          RENT=0
          Not_Verified=0
          Source_Verified=0
          Verified=0
          f=0
          w=0
          year_1=0
          TenPlusYears=0
          years_2=0
          years_3=0
          years_4=0
          years_5=0
          years_6=0
          years_7=0
          years_8=0
          years_9=0
          Lessthan1Year=0
          car=0
          credit_card=0
          debt_consolidation=0
          educational=0
          home_improvement=0
          house=0
          major_purchase=0
          medical=0
          moving=0
          other=0
          renewable_energy=0
          small_business=0
          vacation=0
          wedding=0
          
          if home_ownership[0] == "Any" :
              ANY=1
          if home_ownership[0] == "Mortgage" :
              MORTGAGE=1
          if home_ownership[0] == "None" :
              NONE=1
          if home_ownership[0] == "Other" :
              OTHER=1
          if home_ownership[0] == "Own" :
              OWN=1
          if home_ownership[0] == "Rent" :
              RENT=1
          if verification_status[0] == "Not Verified" :
              Not_Verified=1
          if verification_status[0] == "Source Verified" :
              Source_Verified=1
          if verification_status[0] == "Verified" :
              Verified=1
          if initial_list_status[0] == "Forwarded" :              
              f=1
          if initial_list_status[0] == "Waiting" :
              w=1
          if Experience[0] == "1 year" :
              year_1=1
          if Experience[0] == "10+ years" :
              TenPlusYears=1
          if Experience[0] == "2 years" :
              years_2=1
          if Experience[0] == "2 years" :
              years_3=1
          if Experience[0] == "2 years" :
              years_4=1
          if Experience[0] == "3 years" :
              years_5=1
          if Experience[0] == "2 years" :
              years_6=1
          if Experience[0] == "2 years" :
              years_7=1
          if Experience[0] == "2 years" :
              years_8=1
          if Experience[0] == "2 years" :
              years_9=1
          if Experience[0] == "< 1 year" :
              Lessthan1Year=1
          if purpose[0] == "Car" :
              car=1
          if purpose[0] == "Credit Card" :
              credit_card=1
          if purpose[0] == "Debt Consolidation" :
              debt_consolidation=1
          if purpose[0] == "Educational" :
              educational=1
          if purpose[0] == "Home Improvement" :
              home_improvement=1
          if purpose[0] == "House" :
              house=1
          if purpose[0] == "Major Purpose" :
              major_purchase=1
          if purpose[0] == "Medical" :
              medical=1
          if purpose[0] == "Moving" :
              moving=1
          if purpose[0] == "Other" :
              other=1
          if purpose[0] == "Renewable Energy" :
              renewable_energy=1
          if purpose[0] == "Small Business" :
              small_business=1
          if purpose[0] == "Vacation" :
              vacation=1
          if purpose[0] == "Wedding" :
              wedding=1
          
          new_row={'tot_curr_bal':float(tot_curr_bal[0]),'annual_inc':float(annual_inc[0]),'debt_income_ratio':float(debt_income_ratio[0]),'numb_credit':float(numb_credit[0]),
       'loan_amnt':int(loan_amnt[0]),'total_credits':float(total_credits[0]),'total_rec_int':float(total_rec_int[0]),
       'Rate_of_intrst':float(Rate_of_intrst[0]),'ANY':ANY,'MORTGAGE':MORTGAGE,'NONE':NONE,'OTHER':OTHER,'OWN':OWN,'RENT':RENT,
       'Not Verified':Not_Verified,'Source Verified':Source_Verified,'Verified':Verified,'f':f,'w':w,'1 year':year_1,
       '10PlusYears':TenPlusYears,'2 years':years_2,'3 years':years_3,'4 years':years_4,'5 years':years_5,'6 years':years_6,
       '7 years':years_7,'8 years':years_8,'9 years':years_9,'Lessthan1Year':Lessthan1Year,'car':car,'credit_card':credit_card,
       'debt_consolidation':debt_consolidation,'educational':educational,'home_improvement':home_improvement,'house':house,
       'major_purchase':major_purchase,'medical':medical,'moving':moving,'other':other,'renewable_energy':renewable_energy,
       'small_business':small_business,'vacation':vacation,'wedding':wedding}
          
          pred_df=pd.DataFrame()    
          
          pred_df = pred_df.append(new_row, ignore_index=True)
          
          print(new_row)
          pred_df = pred_df[cols_when_model_builds]
          prediction=model.predict(pred_df)
          output=round(prediction[0],2)
          print(output)
          return render_template('RevolvingBalance.html',Predicted_val=format(output))

       
      
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
    

