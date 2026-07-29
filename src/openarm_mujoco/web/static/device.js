// Copyright 2026 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Shared phone detection, used by both the welcome and register pages.
// No single reliable check exists: UA strings are spoofable and iPadOS
// reports itself as "Macintosh". Combine several signals instead.
function detectDevice() {
  const ua = navigator.userAgent;
  const uaData = navigator.userAgentData;

  const signals = {
    "navigator.userAgentData.mobile":
      uaData && typeof uaData.mobile === "boolean" ? uaData.mobile : null,
    "mobile UA string":
      /Android|iPhone|iPod|Windows Phone|webOS|BlackBerry|Opera Mini|IEMobile/i.test(
        ua,
      ),
    "iPadOS (Mac UA + touch)":
      /Macintosh/.test(ua) && navigator.maxTouchPoints > 1,
    "touch points > 0": navigator.maxTouchPoints > 0,
    "pointer: coarse": window.matchMedia("(pointer: coarse)").matches,
    "DeviceOrientationEvent exists": "DeviceOrientationEvent" in window,
  };

  // Treat as a phone when a mobile identity signal agrees with touch.
  const identity =
    signals["navigator.userAgentData.mobile"] === true ||
    signals["mobile UA string"] ||
    signals["iPadOS (Mac UA + touch)"];
  const isPhone = identity && signals["touch points > 0"];

  return { signals, isPhone };
}
