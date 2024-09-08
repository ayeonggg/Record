#define LEDPIN 11 //LED brightness (PWM) writing
#define LIGHTPIN A0 //Ambient light sensor reading 

void setup() {
 pinMode(LIGHTPIN, INPUT); 
 pinMode(LEDPIN, OUTPUT); 
 Serial.begin(9600);
}

void loop() {
 float reading = analogRead(LIGHTPIN); //Read light level
 float square_ratio = reading / 1023.0; //Get percent of maximum value (1023)
 square_ratio = pow(square_ratio, 2.0); //Square to make response more obvious

 analogWrite(LEDPIN, 255.0 * square_ratio); //Adjust LED brightness relatively
 Serial.println(reading); //Display reading in serial monitor
}
