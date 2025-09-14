FROM node:20-alpine

WORKDIR /app

COPY mcp_server/package*.json ./
RUN npm ci --only=production
RUN apk add --no-cache curl

COPY mcp_server/index.js ./

ENV PORT=5000
EXPOSE 5000

CMD ["node", "index.js"]


