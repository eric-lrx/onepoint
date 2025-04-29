import axios from 'axios';

const API_URL = 'http://localhost:8000';

const apiService = {
  ping: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/ping`);
      return response.data;
    } catch (error) {
      console.error('Error pinging API:', error);
      throw error;
    }
  },
  
  askQuestion: async (question) => {
    try {
      const response = await axios.post(`${API_URL}/api/ask`, null, {
        params: { question }
      });
      return response.data;
    } catch (error) {
      console.error('Error asking question:', error);
      throw error;
    }
  }
};

export default apiService;