#version 330 core

uniform vec4 fColor;
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1;

void main() {
    // FragColor = fColor;
    FragColor = texture(texture1, TexCoord);
}