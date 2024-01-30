def findDecision(obj): #obj[0]: Age, obj[1]: Gender, obj[2]: BMI, obj[3]: Region, obj[4]: No. Childred, obj[5]: Insurance Charges
	# {"feature": "Insurance Charges", "instances": 2836, "metric_value": 0.1209, "depth": 1}
	if obj[5]<=12124.952377244004:
		# {"feature": "Age", "instances": 1928, "metric_value": 0.0877, "depth": 2}
		if obj[0]<=47.84940469874466:
			# {"feature": "BMI", "instances": 1495, "metric_value": 0.0105, "depth": 3}
			if obj[2]<=29.719359774280935:
				# {"feature": "No. Childred", "instances": 818, "metric_value": 0.0037, "depth": 4}
				if obj[4]>2.0:
					# {"feature": "Region", "instances": 568, "metric_value": 0.0162, "depth": 5}
					if obj[3]>0:
						# {"feature": "Gender", "instances": 404, "metric_value": 0.0078, "depth": 6}
						if obj[1]>0:
							return 0
						elif obj[1]<=0:
							return 0
						else: return 0.031007751937984496
					elif obj[3]<=0:
						return 0
					else: return 0.0
				elif obj[4]<=2.0:
					# {"feature": "Region", "instances": 250, "metric_value": 0.0016, "depth": 5}
					if obj[3]<=0:
						# {"feature": "Gender", "instances": 141, "metric_value": 0.0009, "depth": 6}
						if obj[1]>0:
							return 0
						elif obj[1]<=0:
							return 0
						else: return 0.016129032258064516
					elif obj[3]>0:
						# {"feature": "Gender", "instances": 109, "metric_value": 0.0008, "depth": 6}
						if obj[1]<=0:
							return 0
						elif obj[1]>0:
							return 0
						else: return 0.044444444444444446
					else: return 0.03669724770642202
				else: return 0.028
			elif obj[2]>29.719359774280935:
				# {"feature": "No. Childred", "instances": 677, "metric_value": 0.0227, "depth": 4}
				if obj[4]>1.0:
					return 0
				elif obj[4]<=1.0:
					# {"feature": "Gender", "instances": 114, "metric_value": 0.0338, "depth": 5}
					if obj[1]<=0:
						return 0
					elif obj[1]>0:
						# {"feature": "Region", "instances": 47, "metric_value": 0.0297, "depth": 6}
						if obj[3]>0:
							return 0
						elif obj[3]<=0:
							return 0
						else: return 0.0
					else: return 0.02127659574468085
				else: return 0.008771929824561403
			else: return 0.0014771048744460858
		elif obj[0]>47.84940469874466:
			# {"feature": "No. Childred", "instances": 433, "metric_value": 0.0215, "depth": 3}
			if obj[4]<=4.0:
				# {"feature": "BMI", "instances": 415, "metric_value": 0.0028, "depth": 4}
				if obj[2]>27.18066629013058:
					# {"feature": "Gender", "instances": 348, "metric_value": 0.0011, "depth": 5}
					if obj[1]<=0:
						# {"feature": "Region", "instances": 236, "metric_value": 0.0001, "depth": 6}
						if obj[3]<=0:
							return 0
						elif obj[3]>0:
							return 0
						else: return 0.2641509433962264
					elif obj[1]>0:
						# {"feature": "Region", "instances": 112, "metric_value": 0.0003, "depth": 6}
						if obj[3]<=0:
							return 0
						elif obj[3]>0:
							return 0
						else: return 0.20833333333333334
					else: return 0.22321428571428573
				elif obj[2]<=27.18066629013058:
					# {"feature": "Region", "instances": 67, "metric_value": 0.0246, "depth": 5}
					if obj[3]<=0:
						# {"feature": "Gender", "instances": 42, "metric_value": 0.0028, "depth": 6}
						if obj[1]<=0:
							return 0
						elif obj[1]>0:
							return 1
						else: return 0.5833333333333334
					elif obj[3]>0:
						# {"feature": "Gender", "instances": 25, "metric_value": 0.0147, "depth": 6}
						if obj[1]<=0:
							return 0
						elif obj[1]>0:
							return 0
						else: return 0.1
					else: return 0.2
				else: return 0.3880597014925373
			elif obj[4]>4.0:
				# {"feature": "BMI", "instances": 18, "metric_value": 0.2291, "depth": 4}
				if obj[2]>29.72643633:
					return 1
				elif obj[2]<=29.72643633:
					return 0
				else: return 0.0
			else: return 0.9444444444444444
		else: return 0.3094688221709007
	elif obj[5]>12124.952377244004:
		# {"feature": "No. Childred", "instances": 908, "metric_value": 0.0214, "depth": 2}
		if obj[4]<=3.0:
			# {"feature": "Age", "instances": 659, "metric_value": 0.016, "depth": 3}
			if obj[0]>43.87465729135053:
				# {"feature": "BMI", "instances": 341, "metric_value": 0.0078, "depth": 4}
				if obj[2]>31.186491704868033:
					# {"feature": "Gender", "instances": 175, "metric_value": 0.0049, "depth": 5}
					if obj[1]<=0:
						# {"feature": "Region", "instances": 95, "metric_value": 0.0054, "depth": 6}
						if obj[3]<=0:
							return 1
						elif obj[3]>0:
							return 1
						else: return 0.6136363636363636
					elif obj[1]>0:
						# {"feature": "Region", "instances": 80, "metric_value": 0.0, "depth": 6}
						if obj[3]>0:
							return 1
						elif obj[3]<=0:
							return 1
						else: return 0.5483870967741935
					else: return 0.55
				elif obj[2]<=31.186491704868033:
					# {"feature": "Gender", "instances": 166, "metric_value": 0.001, "depth": 5}
					if obj[1]<=0:
						# {"feature": "Region", "instances": 85, "metric_value": 0.0044, "depth": 6}
						if obj[3]<=0:
							return 1
						elif obj[3]>0:
							return 1
						else: return 0.75
					elif obj[1]>0:
						# {"feature": "Region", "instances": 81, "metric_value": 0.0002, "depth": 6}
						if obj[3]<=0:
							return 1
						elif obj[3]>0:
							return 1
						else: return 0.7647058823529411
					else: return 0.7530864197530864
				else: return 0.7771084337349398
			elif obj[0]<=43.87465729135053:
				# {"feature": "Region", "instances": 318, "metric_value": 0.0104, "depth": 4}
				if obj[3]<=0:
					# {"feature": "Gender", "instances": 166, "metric_value": 0.0089, "depth": 5}
					if obj[1]<=0:
						# {"feature": "BMI", "instances": 83, "metric_value": 0.0346, "depth": 6}
						if obj[2]<=37.23293153730344:
							return 1
						elif obj[2]>37.23293153730344:
							return 0
						else: return 0.0
					elif obj[1]>0:
						# {"feature": "BMI", "instances": 83, "metric_value": 0.033, "depth": 6}
						if obj[2]>25.346767789956445:
							return 1
						elif obj[2]<=25.346767789956445:
							return 1
						else: return 0.6666666666666666
					else: return 0.891566265060241
				elif obj[3]>0:
					# {"feature": "Gender", "instances": 152, "metric_value": 0.0043, "depth": 5}
					if obj[1]>0:
						# {"feature": "BMI", "instances": 102, "metric_value": 0.0075, "depth": 6}
						if obj[2]>33.09268778715687:
							return 1
						elif obj[2]<=33.09268778715687:
							return 1
						else: return 0.9772727272727273
					elif obj[1]<=0:
						# {"feature": "BMI", "instances": 50, "metric_value": 0.0951, "depth": 6}
						if obj[2]<=29.9913245784:
							return 1
						elif obj[2]>29.9913245784:
							return 1
						else: return 1.0
					else: return 0.9
				else: return 0.9342105263157895
			else: return 0.8805031446540881
		elif obj[4]>3.0:
			# {"feature": "Age", "instances": 249, "metric_value": 0.0523, "depth": 3}
			if obj[0]<=38.455493201124504:
				# {"feature": "BMI", "instances": 128, "metric_value": 0.0129, "depth": 4}
				if obj[2]>25.7:
					# {"feature": "Gender", "instances": 126, "metric_value": 0.0024, "depth": 5}
					if obj[1]>0:
						# {"feature": "Region", "instances": 125, "metric_value": 0.0, "depth": 6}
						if obj[3]<=0:
							return 0
						else: return 0.264
					elif obj[1]<=0:
						return 0
					else: return 0.0
				elif obj[2]<=25.7:
					return 1
				else: return 1.0
			elif obj[0]>38.455493201124504:
				# {"feature": "BMI", "instances": 121, "metric_value": 0.0065, "depth": 4}
				if obj[2]<=30.171691913719005:
					# {"feature": "Region", "instances": 61, "metric_value": 0.0264, "depth": 5}
					if obj[3]>0:
						# {"feature": "Gender", "instances": 55, "metric_value": 0.0028, "depth": 6}
						if obj[1]>0:
							return 1
						elif obj[1]<=0:
							return 1
						else: return 0.8235294117647058
					elif obj[3]<=0:
						return 1
					else: return 1.0
				elif obj[2]>30.171691913719005:
					# {"feature": "Gender", "instances": 60, "metric_value": 0.0163, "depth": 5}
					if obj[1]>0:
						# {"feature": "Region", "instances": 46, "metric_value": 0.0087, "depth": 6}
						if obj[3]<=0:
							return 1
						elif obj[3]>0:
							return 1
						else: return 0.6
					elif obj[1]<=0:
						# {"feature": "Region", "instances": 14, "metric_value": 0.0454, "depth": 6}
						if obj[3]<=0:
							return 0
						elif obj[3]>0:
							return 1
						else: return 0.6666666666666666
					else: return 0.42857142857142855
				else: return 0.65
			else: return 0.71900826446281
		else: return 0.4899598393574297
	else: return 0.7048458149779736
