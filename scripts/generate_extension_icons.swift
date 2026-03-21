import AppKit
import Foundation

let fileManager = FileManager.default
let rootURL = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
let outputDirectory = rootURL.appendingPathComponent("extension/icons", isDirectory: true)

try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

let backgroundTop = NSColor(srgbRed: 15.0 / 255.0, green: 81.0 / 255.0, blue: 50.0 / 255.0, alpha: 1.0)
let backgroundBottom = NSColor(srgbRed: 45.0 / 255.0, green: 122.0 / 255.0, blue: 70.0 / 255.0, alpha: 1.0)
let foreground = NSColor(srgbRed: 247.0 / 255.0, green: 243.0 / 255.0, blue: 232.0 / 255.0, alpha: 1.0)
let accent = NSColor(srgbRed: 223.0 / 255.0, green: 235.0 / 255.0, blue: 223.0 / 255.0, alpha: 0.95)

func drawRoundedRect(_ rect: CGRect, radius: CGFloat, color: NSColor) {
  color.setFill()
  let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
  path.fill()
}

func drawIcon(size: CGFloat) -> NSImage {
  let imageSize = NSSize(width: size, height: size)
  let image = NSImage(size: imageSize)
  image.lockFocus()

  guard let context = NSGraphicsContext.current?.cgContext else {
    fatalError("Graphics context unavailable")
  }

  context.setAllowsAntialiasing(true)
  context.interpolationQuality = .high
  context.clear(CGRect(origin: .zero, size: imageSize))

  let inset = size * 0.06
  let cardRect = CGRect(x: inset, y: inset, width: size - inset * 2.0, height: size - inset * 2.0)
  let cardPath = NSBezierPath(roundedRect: cardRect, xRadius: size * 0.22, yRadius: size * 0.22)

  NSGraphicsContext.saveGraphicsState()
  cardPath.addClip()
  if let gradient = NSGradient(colors: [backgroundTop, backgroundBottom]) {
    gradient.draw(in: cardPath, angle: -55.0)
  }
  NSGraphicsContext.restoreGraphicsState()

  let innerRect = cardRect.insetBy(dx: size * 0.12, dy: size * 0.14)
  drawRoundedRect(innerRect, radius: size * 0.12, color: NSColor.white.withAlphaComponent(0.08))

  let braceFont = NSFont.monospacedSystemFont(ofSize: size * 0.54, weight: .bold)
  let braceAttributes: [NSAttributedString.Key: Any] = [
    .font: braceFont,
    .foregroundColor: foreground,
  ]

  let leftBrace = NSAttributedString(string: "{", attributes: braceAttributes)
  let rightBrace = NSAttributedString(string: "}", attributes: braceAttributes)

  leftBrace.draw(at: NSPoint(x: size * 0.18, y: size * 0.19))
  rightBrace.draw(at: NSPoint(x: size * 0.60, y: size * 0.19))

  let lineHeight = max(2.0, size * 0.055)
  let lineRadius = lineHeight / 2.0
  let lineX = size * 0.34
  let lineWidths = [0.30, 0.24, 0.20].map { size * $0 }
  let lineYs = [0.62, 0.49, 0.36].map { size * $0 }

  for (index, lineY) in lineYs.enumerated() {
    let color = index == 1 ? accent : foreground
    let rect = CGRect(x: lineX, y: lineY, width: lineWidths[index], height: lineHeight)
    drawRoundedRect(rect, radius: lineRadius, color: color)
  }

  image.unlockFocus()
  return image
}

func writePNG(_ image: NSImage, to url: URL) throws {
  guard
    let tiffData = image.tiffRepresentation,
    let bitmap = NSBitmapImageRep(data: tiffData),
    let pngData = bitmap.representation(using: .png, properties: [:])
  else {
    throw NSError(domain: "GenerateExtensionIcons", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to encode PNG"])
  }

  try pngData.write(to: url)
}

for size in [16, 32, 48, 128] {
  let image = drawIcon(size: CGFloat(size))
  let outputURL = outputDirectory.appendingPathComponent("icon\(size).png")
  try writePNG(image, to: outputURL)
  print("Wrote \(outputURL.path)")
}
