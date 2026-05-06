import { Input } from 'antd'

interface Props {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
}

export default function ScenarioInput({ value, onChange, onSubmit }: Props) {
  return (
    <Input.TextArea
      rows={3}
      placeholder="输入场景描述，如：my colleague is late for an important meeting"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onPressEnter={(e) => {
        if (!e.shiftKey) {
          e.preventDefault()
          onSubmit()
        }
      }}
    />
  )
}