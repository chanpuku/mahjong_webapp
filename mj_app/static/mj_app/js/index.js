function sum_scores() {
	var score=Number($("#id_score0").val())+Number($("#id_score1").val())+Number($("#id_score2").val())
	if($('input[name="num_of_people"]:checked').val()=='4'){
		score+=Number($("#id_score3").val())
	};
	return score;
}

function scores_default(num){
	var target_score=25000;
	if (num==3){
		target_score=35000;
	};
	var scores=$('input[name="score"]')
	for (let i = 0; i < 4; i++) {
		scores[i].value=target_score
	}
};

function change_num_of_kyoku(){
	if ($('input[name="num_of_kyoku"]:checked').val() == "nan") {
		$('#id_bakaze_sha_all').show();
	}else if ($('input[name="num_of_kyoku"]:checked').val() == "ton"){
		$('#id_bakaze_sha_all').hide();
	};
};
function change_kyoku(){
	n=Number($('input[name="num_of_people"]:checked').val());
	k=Number($('input[name="kyoku"]:checked').val());

	winds=['東','南','西','北'];
	for (let i = 0; i < 4; i++) {

		id='#player'+(i+1)+'_label';
		label='Player '+(i+1)+' : '+winds[(i-k+1+n)%n];
		$(id).html(label);
	};
};


//select ボタン
$("#id_rule").change(function(){
	// 選択した値を取得
	var select_value = $(this).val();
	$("#id_bakaze").children().remove();
	$("#id_kyoku").children().remove();
	$("#id_bakaze").children().remove();
	$(".scores").children().remove();
	if (select_value == 1){

	};
});
//ルール設定時の挙動
$('input[name="num_of_people"]').change(function(){
	$('input[name="num_of_kyoku"]')[0].checked = true;
	$('input[name="bakaze"]')[0].checked = true;
	$('input[name="kyoku"]')[0].checked = true;
	change_num_of_kyoku()
	change_kyoku()
	let num=Number($(this).val())
	if (num == 4) {
		$('#id_kyoku_4_all').show();
		$('#id_score3_all').show();
		$('#score_error_message').html('持ち点の合計を100000点にしてください。');
		$('#score_error_message').hide();
		$('#santon_message').hide();
		$('#id_kyoku_tonpu_label').show();
		scores_default(num);
	}else if (num==3){
		$('#id_kyoku_4_all').hide();
		$('#id_score3_all').hide();
		$('#score_error_message').html('持ち点の合計を105000点にしてください。');
		$('#score_error_message').hide();
		$('#santon_message').show();
		$('#id_kyoku_tonpu_label').hide();
		scores_default(num);
	};
});
$('input[name="num_of_kyoku"]').change(function(){
	$('input[name="bakaze"]')[0].checked = true;
	$('input[name="kyoku"]')[0].checked = true;
	change_num_of_kyoku()
	change_kyoku()
});
$('input[name="kyoku"]').change(function(){
	change_kyoku()
});

//点数修正時の挙動
$('input[name="score"]').change(function(){
	let score=sum_scores()
	let num=$('input[name="num_of_people"]:checked').val()
	
	var target_score=0
	if(num=='4'){
		target_score=100000;
	}else if (num=='3'){
		target_score=105000;
	};

	if (score==target_score){
		$('#score_error_message').hide();
	}else{
		$('#score_error_message').show();
	};
});

//実行ボタンの挙動
$('#input_info').on('submit', function(e) {
	e.preventDefault();
	let score=sum_scores()
	let num=$('input[name="num_of_people"]:checked').val()
	if (num=='4'){
		if (score!=100000){return;};
	}else{
		if (score!=105000){return;};
	};

	$('#loading-message').show();
	$.ajax({
		'url':'execute/',
		'type': 'POST',
		'data': {
			'num_of_people': $('input[name="num_of_people"]:checked').val(),
			'num_of_kyoku': $('input[name="num_of_kyoku"]:checked').val(),
			'bakaze': $('input[name="bakaze"]:checked').val(),
			'kyoku': $('input[name="kyoku"]:checked').val(),
			'score0': $('#id_score0').val(),
			'score1': $('#id_score1').val(),
			'score2': $('#id_score2').val(),
			'score3': $('#id_score3').val(),
		},
		'dataType': 'json'
	})

	.done(function(response){
		$('#loading-message').hide();
		$("#result_area").html(response.result);
	});
});

