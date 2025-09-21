import { render, screen } from '@testing-library/react';
import App from './App';
import './index.css'; // or './App.css' if that's where Tailwind is declared


test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
