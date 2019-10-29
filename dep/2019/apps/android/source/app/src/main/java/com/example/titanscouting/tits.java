package com.example.titanscouting;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

public class tits extends AppCompatActivity {
    Button button;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tits);

        button = (Button) findViewById(R.id.button);
        // Capture button clicks
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View arg0) {


                    Intent myIntent = new Intent(tits.this,
                            MainActivity.class);
                    startActivity(myIntent);

            }
        });
    }
}
