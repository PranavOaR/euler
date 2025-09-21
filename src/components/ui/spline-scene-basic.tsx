'use client'

import { SplineScene } from "@/components/ui/splite";
import { Card } from "@/components/ui/card"
import { SpotlightSVG } from "@/components/ui/spotlight-svg"
 
export function SplineSceneBasic() {
  return (
    <Card className="w-full h-[700px] bg-black/[0.96] relative overflow-hidden">
      <SpotlightSVG
        className="-top-40 left-0 md:left-60 md:-top-20"
        fill="white"
      />
      
      <div className="flex h-full">
        {/* Right content - 3D Scene */}
        <div className="w-full relative">
          <SplineScene 
            scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
            className="w-full h-full"
          />
        </div>
      </div>
    </Card>
  )
}