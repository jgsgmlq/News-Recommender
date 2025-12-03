// ==================== 全局变量 ====================
let currentUser = null;
let attentionChart = null;

// ==================== 页面加载 ====================
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadUsers();
    setupEventListeners();
});

// ==================== 事件监听 ====================
function setupEventListeners() {
    // 用户选择
    document.getElementById('userSelect').addEventListener('change', function(e) {
        const userId = e.target.value;
        if (userId) {
            currentUser = userId;
            loadUserHistory(userId);
            document.getElementById('recommendBtn').disabled = false;
        } else {
            currentUser = null;
            document.getElementById('recommendBtn').disabled = true;
            document.getElementById('userHistory').innerHTML = '<div class="empty-state"><p>👈 请先选择用户</p></div>';
        }
    });

    // 推荐按钮
    document.getElementById('recommendBtn').addEventListener('click', function() {
        if (currentUser) {
            generateRecommendations(currentUser);
        }
    });

    // 模态框关闭
    document.querySelector('.close').addEventListener('click', closeModal);
    window.addEventListener('click', function(e) {
        const modal = document.getElementById('detailModal');
        if (e.target === modal) {
            closeModal();
        }
    });
}

// ==================== API调用 ====================
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        document.getElementById('totalNews').textContent = data.total_news.toLocaleString();
        document.getElementById('totalUsers').textContent = data.total_users.toLocaleString();
        document.getElementById('modelStatus').textContent = data.model_loaded ? '✅ 已加载' : '❌ 未加载';
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadUsers() {
    try {
        const response = await fetch('/api/users');
        const data = await response.json();

        const select = document.getElementById('userSelect');
        data.users.forEach(user => {
            const option = document.createElement('option');
            option.value = user.user_id;
            option.textContent = `${user.user_id} (${user.history_count} 条历史)`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading users:', error);
    }
}

async function loadUserHistory(userId) {
    try {
        const response = await fetch(`/api/user/${userId}/history`);
        const data = await response.json();

        const historyDiv = document.getElementById('userHistory');

        if (data.history.length === 0) {
            historyDiv.innerHTML = '<div class="empty-state"><p>该用户暂无历史记录</p></div>';
            return;
        }

        let html = '';
        data.history.forEach(news => {
            html += `
                <div class="news-card">
                    <span class="news-category">${news.category}</span>
                    <div class="news-title">${news.title}</div>
                    <div class="news-abstract">${news.abstract}</div>
                </div>
            `;
        });

        historyDiv.innerHTML = html;
    } catch (error) {
        console.error('Error loading user history:', error);
        document.getElementById('userHistory').innerHTML = '<div class="empty-state"><p>❌ 加载失败</p></div>';
    }
}

async function generateRecommendations(userId) {
    // 显示loading
    showLoading();

    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: userId,
                top_k: 10
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayRecommendations(data.recommendations);
    } catch (error) {
        console.error('Error generating recommendations:', error);
        document.getElementById('recommendations').innerHTML = `
            <div class="empty-state">
                <p>❌ 推荐失败: ${error.message}</p>
            </div>
        `;
    } finally {
        hideLoading();
    }
}

// ==================== UI渲染 ====================
function displayRecommendations(recommendations) {
    const recDiv = document.getElementById('recommendations');

    if (recommendations.length === 0) {
        recDiv.innerHTML = '<div class="empty-state"><p>暂无推荐结果</p></div>';
        return;
    }

    let html = '';
    recommendations.forEach(rec => {
        const attnWeights = rec.attention_weights;
        html += `
            <div class="recommend-card" onclick="showDetail(${JSON.stringify(rec).replace(/"/g, '&quot;')})">
                <div class="recommend-rank">${rec.rank}</div>
                <span class="recommend-score">匹配度: ${(rec.score * 100).toFixed(1)}%</span>
                <span class="news-category">${rec.category}</span>
                <div class="news-title">${rec.title}</div>
                <div class="news-abstract">${rec.abstract}</div>
                <div class="attention-preview">
                    <span class="attention-tag">ID: ${(attnWeights.id * 100).toFixed(0)}%</span>
                    <span class="attention-tag">LLM: ${(attnWeights.llm * 100).toFixed(0)}%</span>
                    <span class="attention-tag">GNN: ${(attnWeights.gnn * 100).toFixed(0)}%</span>
                </div>
            </div>
        `;
    });

    recDiv.innerHTML = html;
}

function showDetail(recommendation) {
    const modal = document.getElementById('detailModal');
    document.getElementById('modalTitle').textContent = recommendation.title;

    // 新闻信息
    const newsInfo = document.getElementById('newsInfo');
    newsInfo.innerHTML = `
        <p><strong>类别:</strong> ${recommendation.category} / ${recommendation.subcategory}</p>
        <p><strong>匹配度:</strong> ${(recommendation.score * 100).toFixed(2)}%</p>
        <p><strong>排名:</strong> #${recommendation.rank}</p>
        <p><strong>摘要:</strong> ${recommendation.abstract}</p>
    `;

    // 注意力权重图表
    renderAttentionChart(recommendation.attention_weights);

    // 推荐原因
    const reason = generateRecommendationReason(recommendation);
    document.getElementById('recommendReason').innerHTML = reason;

    modal.style.display = 'block';
}

function renderAttentionChart(weights) {
    const ctx = document.getElementById('attentionChart').getContext('2d');

    // 销毁旧图表
    if (attentionChart) {
        attentionChart.destroy();
    }

    const data = {
        labels: ['ID嵌入', 'LLM嵌入', 'GNN嵌入'],
        datasets: [{
            label: '注意力权重',
            data: [weights.id, weights.llm, weights.gnn],
            backgroundColor: [
                'rgba(255, 99, 132, 0.7)',
                'rgba(54, 162, 235, 0.7)',
                'rgba(75, 192, 192, 0.7)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(75, 192, 192, 1)'
            ],
            borderWidth: 2
        }]
    };

    attentionChart = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });

    // 解释文本
    const explanation = document.getElementById('attentionExplanation');
    const maxWeight = Math.max(weights.id, weights.llm, weights.gnn);
    let dominantModality = 'ID嵌入';
    if (weights.llm === maxWeight) dominantModality = 'LLM嵌入';
    if (weights.gnn === maxWeight) dominantModality = 'GNN嵌入';

    explanation.innerHTML = `
        <strong>权重分析：</strong>
        <br>本次推荐主要依赖 <strong>${dominantModality}</strong> (${(maxWeight * 100).toFixed(1)}%)。
        <br><br>
        <strong>模态说明：</strong><br>
        • <strong>ID嵌入</strong>: 协同过滤信号，反映群体点击偏好<br>
        • <strong>LLM嵌入</strong>: 语义理解信号，捕捉新闻内容深层含义<br>
        • <strong>GNN嵌入</strong>: 知识图谱信号，利用实体关联推荐
    `;
}

function generateRecommendationReason(rec) {
    const weights = rec.attention_weights;
    let reason = '<p>';

    // 基于权重生成推荐原因
    if (weights.llm > 0.5) {
        reason += `📚 <strong>语义匹配度高</strong>：该新闻的语义内容与您的阅读偏好高度契合。LLM嵌入权重达到 <strong>${(weights.llm * 100).toFixed(1)}%</strong>，说明新闻主题、观点和表达方式与您历史阅读的文章相似。<br><br>`;
    }

    if (weights.id > 0.4) {
        reason += `👥 <strong>群体推荐</strong>：与您兴趣相似的用户也经常点击此类新闻。ID嵌入权重为 <strong>${(weights.id * 100).toFixed(1)}%</strong>，反映了协同过滤的群体智慧。<br><br>`;
    }

    if (weights.gnn > 0.3) {
        reason += `🔗 <strong>实体关联</strong>：该新闻提到的实体（人物、组织、地点）与您感兴趣的主题相关。GNN嵌入权重为 <strong>${(weights.gnn * 100).toFixed(1)}%</strong>，通过知识图谱发现了潜在关联。<br><br>`;
    }

    reason += `🎯 <strong>综合匹配度</strong>：${(rec.score * 100).toFixed(2)}%，在所有候选新闻中排名第 <strong>${rec.rank}</strong> 位。`;
    reason += '</p>';

    return reason;
}

function closeModal() {
    document.getElementById('detailModal').style.display = 'none';
}

function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}
