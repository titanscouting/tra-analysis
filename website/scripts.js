var currentPageNum=0;
var contentArray=[];
$(document).ready(function(){
    $("#add").click(function(){
        var x = $('.formNavi').length;
        $("#add").before("<div class='formNavi'><input type='text' name='' value='Page "+x+"'></div>");
        $("#add").text="+";
        contentArray.push([]);
        currentPageNum=x-1;
    });
});
$(document).ready(function(){
    $("#RemPage").click(function(){
      if($(".formNavi").length>1){
        $(".formNavi:nth-child("+(currentPageNum+1)+")").remove();
        $("#add").text="+";
        contentArray.splice(currentPageNum,1);
        currentPageNum--;
      }
    });
});
$(document).ready(function(){
    $("#newQ").click(function(){
        $("#svb").before("<div class ='question'>");
        $("#add").text="+";

    });
});
