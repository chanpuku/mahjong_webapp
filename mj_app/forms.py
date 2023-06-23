from django import forms


rule_list=( ('1','四麻東風戦'),
		('2','四麻半荘戦'),
		('3','三麻東風戦'),
		('4','三麻半荘戦'))

bakaze_list=(('1','東'),
			('2','南'),
			('3','西'))

kyoku_list=( ('1','1'),
			('2','2'),
			('3','3'),
			('4','4'))

class DataForm(forms.Form):
	rule = forms.ChoiceField(required=True,choices=rule_list)
	bakaze = forms.ChoiceField(required=True,choices=bakaze_list)
	kyoku = forms.ChoiceField(required=True,choices=kyoku_list)
	score0 = forms.IntegerField(label='東',required=True,initial=35000,max_value=105000,min_value=0)
	score1 = forms.IntegerField(label='南',required=True,initial=35000,max_value=105000,min_value=0)
	score2 = forms.IntegerField(label='西',required=True,initial=35000,max_value=105000,min_value=0)
	score3 = forms.IntegerField(label='北',required=True,initial=35000,max_value=105000,min_value=0)

	