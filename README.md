# quantsim-remote

llama-3.2-3B 양자화 시뮬레이터(quantsim-hub)의 **고정 원격 접속 주소**.

이 저장소는 GitHub Pages로 서빙되는 정적 런처 페이지 하나(`index.html`)로만
구성된다. 실제 애플리케이션(Python 백엔드)은 여기 없고, 접속 시 Tailscale
Funnel 주소로 리다이렉트만 한다.

- 이 GitHub Pages 주소는 **절대 바뀌지 않는다** — 모바일 홈 화면에 추가해두면
  영구 북마크로 쓸 수 있다.
- 실제 서버(Tailscale Funnel URL) 쪽이 바뀌면 `index.html` 의 `HUB_URL` 상수만
  고쳐서 재배포하면 된다.
- 접속 토큰은 **소스에 포함하지 않는다** — GitHub Pages 는 저장소 공개 여부와
  무관하게 결과물이 공개 웹이므로, 하드코딩하면 토큰이 그대로 노출된다.
  대신 각 기기에서 최초 1회 입력 → `localStorage` 저장 방식을 쓴다.
- 서버(로컬 PC)가 꺼져 있으면 이 페이지는 열리지만 리다이렉트 후 접속은
  실패한다 — 이건 이 저장소의 한계가 아니라 원격 PC 자체가 꺼져 있다는 뜻.
