const express = require('express');
const app = express();
app.use(express.json());  // JSON 바디 파싱 활성화

// MCP 기본 테스트 엔드포인트
app.post('/mcp/context', (req, res) => {
    console.log('✅ MCP 수신 데이터:', req.body);
    res.json({ status: 'success', message: 'MCP 컨텍스트 처리 완료' });
});

const PORT = 5000;
app.listen(PORT, () => console.log(`🚀 MCP 서버 실행: http://localhost:${PORT}`)); // 추후, EC2 공용 IP로 대체하여 배포, 5000 PORT 열어야함!