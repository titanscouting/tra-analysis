package com.example.titanscouting;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.app.Activity;
import android.content.Intent;
import android.view.Menu;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;
public class launcher extends AppCompatActivity {

    Button button;
    EditText passField;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_launcher);

        // Locate the button in activity_main.xml
        button = (Button) findViewById(R.id.launch_button);
        final EditText passField = (EditText)findViewById(R.id.editText);
        // Capture button clicks
        button.setOnClickListener(new OnClickListener() {
            public void onClick(View arg0) {

                // Start NewActivity.class
                if(passField.getText().toString().equals("gimmetits")){

                    Intent myIntent = new Intent(launcher.this,
                            tits.class);
                    startActivity(myIntent);

                }
                else {
                    Intent myIntent = new Intent(launcher.this,
                            MainActivity.class);
                    startActivity(myIntent);
                }
            }
        });

    }



}
