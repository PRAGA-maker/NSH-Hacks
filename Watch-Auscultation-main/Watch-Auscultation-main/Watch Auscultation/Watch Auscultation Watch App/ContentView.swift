//
//  ContentView.swift
//  Watch Auscultation Watch App
//
//  Created by Rucha Nandha on 11/4/23.
//

import SwiftUI


struct AnimationValues {
        var scale = 1.0
        var verticalStretch = 1.0
        var verticalTranslation = 0.0
        var angle = Angle.zero
    }

struct ContentView: View {
    @Binding var isRecording: Bool
    
    var body: some View {
        NavigationStack {
            VStack {
                Spacer()
                Image(systemName: "heart.fill")
                    .foregroundStyle(.red)
                    .font(.largeTitle)
                    .keyframeAnimator(initialValue: AnimationValues()) { content, value in
                        content
                            .foregroundStyle(.white)
                            .rotationEffect(value.angle)
                            .scaleEffect(value.scale)
                            .scaleEffect(y: value.verticalStretch)
                            .offset(y: value.verticalTranslation)
                    } keyframes: { _ in
                        KeyframeTrack(\.angle) {
                            CubicKeyframe(.zero, duration: 0.58)
                            CubicKeyframe(.degrees(16), duration: 0.125)
                            CubicKeyframe(.degrees(-16), duration: 0.125)
                            CubicKeyframe(.degrees(16), duration: 0.125)
                            CubicKeyframe(.zero, duration: 0.125)
                        }
                        
                        KeyframeTrack(\.verticalStretch) {
                            CubicKeyframe(1.0, duration: 0.1)
                            CubicKeyframe(0.6, duration: 0.15)
                            CubicKeyframe(1.5, duration: 0.1)
                            CubicKeyframe(1.05, duration: 0.15)
                            CubicKeyframe(1.0, duration: 0.88)
                            CubicKeyframe(0.8, duration: 0.1)
                            CubicKeyframe(1.04, duration: 0.4)
                            CubicKeyframe(1.0, duration: 0.22)
                        }
                        
                        KeyframeTrack(\.scale) {
                            LinearKeyframe(1.0, duration: 0.36)
                            SpringKeyframe(1.5, duration: 0.8, spring: .bouncy)
                            SpringKeyframe(1.0, spring: .bouncy)
                        }
                        
                        KeyframeTrack(\.verticalTranslation) {
                            LinearKeyframe(0.0, duration: 0.1)
                            SpringKeyframe(20.0, duration: 0.15, spring: .bouncy)
                            SpringKeyframe(-60.0, duration: 1.0, spring: .bouncy)
                            SpringKeyframe(0.0, spring: .bouncy)
                        }
                        
                    }
                
                Spacer()
                NavigationLink("Record", destination: RecordScreen(isRecording: $isRecording))
                
            }
            
                
                
            
        }
    }
}

#Preview {
    ContentView(isRecording: .constant(true))
       
}
